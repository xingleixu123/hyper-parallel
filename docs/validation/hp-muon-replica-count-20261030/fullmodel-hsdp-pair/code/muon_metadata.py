"""Read existing Muon assignment metadata; no optimizer or device operation."""
from collections.abc import Mapping


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _param_bindings(base):
    """Bind actual main-param identities to actual model FQNs, never infer from names."""
    pairs = list(base.model.named_parameters())
    _require(pairs, "model has no named parameters")
    _require(len({name for name, _ in pairs}) == len(pairs), "duplicate model FQN")
    _require(len({id(param) for _, param in pairs}) == len(pairs), "duplicate model identity")
    mapping = getattr(base.optimizer, "model_param_by_optimizer_param", None)
    _require(isinstance(mapping, Mapping), "missing optimizer-to-model identity map")
    reverse = {}
    for main, model in mapping.items():
        _require(id(main) not in reverse, "duplicate optimizer main identity")
        reverse[id(main)] = (main, model)
    result = {}
    for fqn, model in pairs:
        _require(isinstance(fqn, str) and fqn, "invalid model FQN")
        main = getattr(model, "main_param", None)
        _require(main is not None and id(main) not in result, "missing/duplicate FP32 main")
        _require(str(main.dtype) == "torch.float32", "non-FP32 main: " + fqn)
        if main is not model:
            bound = reverse.get(id(main))
            _require(bound is not None and bound[0] is main and bound[1] is model,
                     "independent main identity differs: " + fqn)
        elif id(main) in reverse:
            bound = reverse[id(main)]
            _require(bound[0] is main and bound[1] is model, "alias identity differs: " + fqn)
        for param in (model, main):
            name = getattr(param, "model_name", None)
            _require(name is None or name == fqn, "model_name disagrees with identity: " + fqn)
        result[id(main)] = {"fqn": fqn, "model": model, "main": main}
    for main, model in mapping.items():
        entry = result.get(id(main))
        _require(entry is not None and entry["main"] is main and entry["model"] is model
                 and getattr(model, "main_param", None) is main, "extra or invalid reverse identity map")
    return result


def _layout(param, rank):
    """Read the existing HP DTensor layout, never call DeviceMesh.get_group()."""
    mesh = getattr(param, "device_mesh", None)
    placements = getattr(param, "placements", None)
    _require(mesh is not None and placements is not None and len(placements) == 2, "missing R2S2 layout")
    shape = [int(value) for value in mesh.shape]
    ranks = [int(value) for value in mesh.rank_list]
    coord = [int(value) for value in mesh.get_coordinate()]
    _require(shape == [2, 2] and len(ranks) == 4 and set(ranks) == set(range(4)), "mesh is not four-rank R2S2")
    _require(len(coord) == 2 and all(value in (0, 1) for value in coord), "invalid R2S2 coordinate")
    _require(ranks[2 * coord[0] + coord[1]] == rank, "mesh coordinate/rank mismatch")
    _require(placements[0].is_replicate() and placements[1].is_shard(), "expected replicate then shard")
    global_shape = [int(value) for value in param.shape]
    dim = getattr(placements[1], "dim", None)
    _require(type(dim) is int and 0 <= dim < len(global_shape), "invalid shard dimension")
    local = param.to_local()  # HP DTensor.to_local returns its existing _local_tensor.
    names = getattr(mesh, "mesh_dim_names", None)
    return {
        "global_shape": global_shape,
        "local_shape": [int(value) for value in local.shape],
        "dtype": str(param.dtype),
        "mesh_shape": shape,
        "mesh_rank_list": ranks,
        "mesh_coordinate": coord,
        "mesh_dimension_names": list(names) if names is not None else None,
        "placements": [{"class": type(value).__module__ + "." + type(value).__qualname__,
                        "repr": repr(value), "dim": getattr(value, "dim", None),
                        "uneven_shard": getattr(value, "uneven_shard", None)} for value in placements],
        "mesh_replica_ranks_derived": [ranks[coord[1]], ranks[2 + coord[1]]],
        "mesh_shard_ranks_derived": [ranks[2 * coord[0]], ranks[2 * coord[0] + 1]],
    }


def _bound_layout(entry, rank):
    main_layout = _layout(entry["main"], rank)
    model_layout = _layout(entry["model"], rank)
    for key in ("global_shape", "local_shape", "mesh_shape", "mesh_rank_list", "mesh_coordinate",
                "mesh_dimension_names", "placements"):
        _require(main_layout[key] == model_layout[key], "model/main layout differs: " + entry["fqn"])
    return {"main_is_model": entry["main"] is entry["model"],
            "model_name_attribute": getattr(entry["main"], "model_name", None),
            "main": main_layout, "model": model_layout}


def _pg_ranks(pg, rank, torch_module):
    _require(pg is not None, "missing existing assignment process group")
    ranks = [int(value) for value in torch_module.distributed.get_process_group_ranks(pg)]
    _require(ranks and len(ranks) == len(set(ranks)) and rank in ranks, "invalid assignment PG ranks")
    return ranks


def _state_metadata(leaf, param):
    """Describe raw leaf.state without creating, broadcasting, or cleaning slots."""
    values = leaf.state.get(param)
    if values is None:
        return {"state_entry_present": False, "slot_names": [], "slot_types": {},
                "momentum_buffer_present": False}
    _require(isinstance(values, Mapping) and all(isinstance(key, str) for key in values), "invalid raw Muon state")
    return {"state_entry_present": True, "slot_names": sorted(values),
            "slot_types": {key: type(values[key]).__module__ + "." + type(values[key]).__qualname__
                           for key in sorted(values)},
            "momentum_buffer_present": "momentum_buffer" in values}


def observe_muon(base, cfg, rank, torch_module, *, muon_type=None):
    """Return rank-local assignment metadata for the two frozen config arms.

    muon_type is a test seam; the real caller uses the actual runtime class.
    No collectives, device synchronization, state_dict, or tensor updates occur.
    """
    _require(type(rank) is int and 0 <= rank < 4, "invalid rank")
    if muon_type is None:
        from hyper_parallel.core.optimizer.muon import Muon as muon_type
    leaves = [leaf for leaf in base.optimizer.chained_optimizers if isinstance(leaf, muon_type)]
    _require(len(leaves) == 1, "expected exactly one Muon leaf")
    leaf = leaves[0]
    options = cfg["optimizer"]["muon_config"]
    _require(isinstance(options, Mapping), "missing Muon config")
    key = "muon_hsdp_replica_count"
    present = key in options
    configured = options.get(key)
    _require((not present and configured is None) or (present and type(configured) is int and configured == 1),
             "unsupported configured replica count")
    explicit = getattr(leaf, "_hsdp_replica_count_is_explicit", None)
    effective = getattr(leaf, "hsdp_replica_count", "missing")
    _require(type(explicit) is bool and explicit == present, "configured/explicit flag mismatch")
    _require((not present and effective is None) or (present and type(effective) is int and effective == 1),
             "configured/effective count mismatch")
    bindings = _param_bindings(base)
    groups = leaf.param_groups
    assignment_groups = getattr(leaf, "_hsdp_assignment_batches", None)
    _require(isinstance(assignment_groups, Mapping) and set(assignment_groups) == set(range(len(groups))),
             "Muon assignment group coverage differs")
    seen, muon_ids = set(), set()
    output_groups = []
    for group_index, group in enumerate(groups):
        params = list(group["params"])
        _require(len({id(param) for param in params}) == len(params), "duplicate parameter in Muon group")
        for param in params:
            entry = bindings.get(id(param))
            _require(entry is not None and entry["main"] is param, "Muon parameter lacks identity binding")
            _require(id(param) not in muon_ids, "Muon parameter belongs to multiple groups")
            muon_ids.add(id(param))
        info = assignment_groups[group_index]
        _require(isinstance(info, Mapping), "invalid Muon assignment group")
        output = {"group_index": group_index, "no_comm": [], "assignments": []}
        for param in info.get("no_comm", []):
            ident = id(param)
            _require(any(item is param for item in params) and ident not in seen, "invalid no-comm binding")
            entry = bindings[ident]
            output["no_comm"].append({"fqn": entry["fqn"], "current_owned": True,
                                      "binding_and_layout": _bound_layout(entry, rank), "raw_state": _state_metadata(leaf, param)})
            seen.add(ident)
        for batch_index, batch in enumerate(info.get("batch_groups", [])):
            for sub_index, assignment in enumerate(batch["sub_batches"]):
                replica_pgs = [_pg_ranks(pg, rank, torch_module) for pg in assignment.replicate_pgs]
                shard_pgs = [_pg_ranks(pg, rank, torch_module) for pg in assignment.shard_pgs]
                _require(len(replica_pgs) == 1 and len(shard_pgs) == 1, "expected one replica and one shard PG")
                sizes = [int(value) for value in assignment.replicate_sizes]
                current_coord = [int(value) for value in assignment.replicate_group_ranks]
                _require(sizes == ([1] if present else [2]) and len(current_coord) == 1,
                         "assignment replica sizes differ")
                actual_owned = {id(record.param) for record in assignment.owned_records}
                _require(len(actual_owned) == len(assignment.owned_records), "duplicate owned assignment record")
                records = []
                for record in assignment.all_records:
                    index, param = record.index, record.param
                    _require(type(index) is int and 0 <= index < len(params) and params[index] is param,
                             "assignment record index/identity differs")
                    ident = id(param)
                    entry = bindings.get(ident)
                    _require(entry is not None and entry["main"] is param and ident not in seen,
                             "assignment record has no unique FQN identity")
                    owner = [int(value) for value in assignment.owner_rank_coord(record)]
                    owned = bool(assignment.is_owned(record))
                    _require(len(owner) == 1 and 0 <= owner[0] < sizes[0]
                             and owned == (owner == current_coord) == (ident in actual_owned),
                             "owner coordinate/owned-record mismatch")
                    if present:
                        _require(owned, "count1 assignment not owned locally")
                    bound = _bound_layout(entry, rank)
                    layout = bound["main"]
                    _require(current_coord == ([0] if present else [layout["mesh_coordinate"][0]]),
                             "assignment current coordinate differs from mesh")
                    _require(set(shard_pgs[0]) == set(layout["mesh_shard_ranks_derived"]), "shard PG differs from mesh")
                    expected_replica = {rank} if present else set(layout["mesh_replica_ranks_derived"])
                    _require(set(replica_pgs[0]) == expected_replica, "replica PG differs from configured mesh")
                    _require(type(assignment.is_shard) is bool, "invalid NS shard classification")
                    records.append({"fqn": entry["fqn"], "record_index": index, "owner_coord": owner,
                                    "current_owned": owned, "is_shard": assignment.is_shard,
                                    "replica_pgs": replica_pgs, "shard_pgs": shard_pgs,
                                    "binding_and_layout": bound, "raw_state": _state_metadata(leaf, param)})
                    seen.add(ident)
                _require(records and actual_owned <= {id(record.param) for record in assignment.all_records},
                         "empty assignment or unexpected owned record")
                output["assignments"].append({"batch_group_index": batch_index, "sub_batch_index": sub_index,
                                              "replicate_sizes": sizes, "replicate_coordinate": current_coord,
                                              "records": records})
        output_groups.append(output)
    _require(seen == muon_ids, "Muon records/no-comm do not cover all group parameters")
    for param in leaf.state:
        _require(id(param) in muon_ids and bindings[id(param)]["main"] is param, "unbound raw Muon state")
    return {"schema": "muon-assignment-metadata-v1", "rank": rank,
            "configured_key_present": present, "configured_replica_count": configured,
            "explicit_flag": explicit, "effective_replica_count": effective,
            "muon_parameter_count": len(muon_ids), "groups": output_groups,
            "scope": "existing assignment/PG metadata and raw state presence only; no execution or collective trace"}


def link_momentum_hashes(observation, tensors):
    """Join already captured local hashes to the actual raw owner map, without tensor reads."""
    slots = tensors["optimizer_local_slots"]
    _require(isinstance(slots, Mapping), "invalid captured slot map")
    result = {}
    for group in observation["groups"]:
        records = list(group["no_comm"])
        for assignment in group["assignments"]:
            records.extend(assignment["records"])
        for record in records:
            fqn = record["fqn"]
            _require(fqn not in result, "duplicate Muon FQN in hash join")
            key = fqn + ".momentum_buffer"
            present = record["raw_state"]["momentum_buffer_present"]
            slot = slots.get(key)
            _require((slot is not None) == present, "raw momentum presence/hash coverage differs: " + fqn)
            if present:
                _require(isinstance(slot, Mapping) and isinstance(slot.get("local_sha256"), str)
                         and len(slot["local_sha256"]) == 64
                         and all(char in "0123456789abcdef" for char in slot["local_sha256"]),
                         "invalid captured momentum hash: " + fqn)
            result[fqn] = {"current_owned": record["current_owned"],
                           "momentum_present": present, "momentum_record": slot}
    _require(len(result) == observation["muon_parameter_count"], "incomplete Muon hash join")
    return result
