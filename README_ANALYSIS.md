#### Data files

- `lookup_dict.p`: Dictionary mapping particle IDs to arrays describing their multiple system properties. The columns are:
  - `LOOKUP_SNAP`: Snapshot number.
  - `LOOKUP_PID`: Particle ID.
  - `LOOKUP_SID`: Parent system index.
  - `LOOKUP_MULT`: Particle multiplicity.
  - `LOOKUP_MTOT`: Mass including gas halo [Msun].
  - `LOOKUP_M`: Mass excluding gas halo [Msun].
  - `LOOKUP_SMA`: Semi-major axis [pc].
  - `LOOKUP_ECC`: Eccentricity.
  - `LOOKUP_Q`: Mass ratio.
  - `LOOKUP_MPRIM`: Primary mass [Msun].
  - `LOOKUP_MPRIM_ID`: Primary star index.
  - `LOOKUP_STAR_ORDER`: Rank in multiple (0 = Primary).
  - Note: If single, `LOOKUP_SMA`, `LOOKUP_ECC`, and `LOOKUP_Q` are -1.
- `unique_bin_ids_mult.npz`: Unique binary IDs throughout the simulation, including those in higher-order multiples.
- `fst_mult.npz`: Initial snapshot data for all binary pairs.
- `path_lookup.p`: Pickle file containing particle positions and masses. The columns are:
  - `t`: Snapshot number
  - `id`: Particle ID. 
  - `px, py, pz`: Position components [pc]. 
  - `vx, vy, vz`: Velocity components [1 m/s].
  - `h`: Softening length [pc]. 
  - `m`: Mass of particle (without halo) [Msun].
  - `sys_id`: System index (in systems list)
  - `mtot`: Mass of particle + halo [Msun].
  - `sma`: Semi-major axis [pc].
  - `ecc`: Eccentricity.
  - Note: Unlike lookup, all arrays have the same number of rows. We use np.inf as placeholders if stars do not exist at a particular snapshot.

- `dat_coll_mult.npz`: Contains various binary properties:
  - `bin_ids`: Binary pairs.
  - `fst`: Initial snapshot together (IST) for binaries.
  - `ens`: -PE/KE at IST (without gas).
  - `ens_gas`: -PE/KE at IST (with gas).
  - `same_sys_at_fst`: Whether pair is in same multiple at IST.
  - `bin_at_fst`: Whether pair is a binary at IST.
  - `vangs`: Angle between velocity vectors.
  - `vangs_prim`: Angle between separation and secondary velocity.
  - `mfinal_primary`: Max primary mass at last snapshot both exist.
  - `mfinal_pair`: Total pair mass at last snapshot both exist [Msun].
  - `quasi_filter`: Mask for selecting persistent binaries [Msun].
  - `init_bound_snaps`: First snapshot pair is binary.
  - `final_bound_snaps`: Last snapshot pair is binary.
  - `bound_time`: Number of snapshots pair remains bound.
  - `delta_snap`: Time difference between star formation.
  - `same_sys_final_norm`: Whether pair remains in same system at the end. CAUTION: Do not
            use this filter to check survival, as it does not require the multiple to be perisistent
  - `final_pair_mass_lbin`: Total pair mass at last binary snapshot [Msun].
  - `final_pair_mass_lsys`: Total pair mass at last system snapshot [Msun].
  - `snap_interval`: Time interval between snapshots.
  - Note: Mass fields *exclude* gas halos.


- `mults[_flat].pq`: Pandas DataFrame with multiple systems and their longevity. Stores all high order multiples and sub-systems.
[The flat suffix refers to the fact that each multiple's just depends on the stars it contains and not the hierarchy -- this behavior is controlled by the flat_id parameter.]
  - Columns: `id`, `t`, `tf`, `a`, `e`, `p`, `ss`, `frac_of_orbit`, `nbound_snaps`, `frac_of_orbit_seg`, `nbound_snaps_seg`.
  - `id`: Particle IDs in multiple (Index). 
  - `t`: Snapshot number (Index).
  - `tf`: Final snapshot number
  - `a`: Semi-major axis [pc]
  - `e`: Eccentricity
  - `p`: Period [yr]
  - `ss`: Whether system is a sub-system of another.
  - `frac_of_orbit`: Fraction of orbit systems remains bound 
  - `nbound_snaps`: Number of snapshots system remains bound.
  - `frac_of_orbit_seg`: Fraction of orbit systems remains bound *within current, contiguous time segment*
  - `nbound_snaps_seg`: Number of snapshots system remains bound *within current, contiguous time segment*
  
- `fates_corr*.npz`:
  - `end_states`: Final multiplicity of binary pairs.
  - `same_sys_filt`: Whether pairs remain in persistent multiples. THIS is what is used to select survivors

- `pmult_before_bin*.npz`: Archive containing data about exchanges. In particular the following arrays.
  - `pmult_filt`: Bool array indicating whether exchange occurred before the initial binary formation.
  - `ex_time`: Start time of first exchange.
  - `ex_time_end`: End time of first exchange.
  - `ex_time_max`: Start time of last exchange
  - `ex_time_max_end`: End time of last exchange

- `companions*.p`: Dictionary containing companions for binary stars. For efficiency I have only included data on persistent binaries. It contains 
  - `comp_a_ids`: List of 1st star's companions at `comp_a_times` (with hierarchy including star)
  - `comp_b_ids`: List of 2nd star's companions at `comp_b_times` (with heirarchy including star)
  - `comp_a_ids_flat` and `comp_b_ids_flat`: Flattened lists of companions at these times.

