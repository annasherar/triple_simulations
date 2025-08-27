### Data for n-body integrations

I have organized particle data for candidate exchanges into two pandas dataframes: `data/init_state_all.pq` (the state at 
the start of the exchange) and `data/end_state_all.pq` (the state at the end). You will use a subset of this data to setup your N-body integrations.
This is the data for the last exchange from each binary.

  - Columns:  `t`, `id`, `x`, `y`, `z`, `vx`, `vy`, `vz`, `mcol`, `mtotcol`, `hcol`, `is_soft`.
  - `t`: Snapshot number.
  - `id`: Particle ID. 
  - `x`: x position [pc]
  - `y`: y position [pc]
  - `z`: z position [pc]
  - `vx`: x velocity [m/s]
  - `vy`: y velocity [m/s]
  - `vz`: z velocity [m/s]
  - `mcol`: Particle mass [Msun]
  - `mtotcol`: Particle mass + its gas halo [Msun]
  - `hcol`: Softening length [pc]
  - `is_soft`: Flag for particles affected by softening (0 if False, 1 if True)

The index is the binary id. Each binary id has several rows: The first two in each index are the binary pair. The other 
rows give data on companions (FROM THE START OF THE EXCHANGE. These may no longer be companions at the end but their 
positions are recorded in end_states_all in any case. There can be new companions at the end in principle, but this data is
not included for simplicity, and we will filter out such cases for now). The index should match the one from the dashboard.

There are a couple of special columns in each dataframe that are useful for filtering. 

## `init_state_all.pq`
  - Columns:  `no_new_comps`, `frac_halo`, `frac_halo_max`.
  - `no_new_comps`: Flag indicating no new stars appeared as companions between the initial and final states.
  - `frac_halo`: Fractional mass of the halo for each stars
  - `frac_halo_max`: Maximum of frac_halo for each index.

## `end_state_all.pq`
  - Columns:  `bdist`,	`sma`,	`ecc`.
  - `bdist`: Distance between binary pair [pc]
  - `sma`: Semi-major axis of the binary pair [pc]
  - `ecc`: Eccentricity of binary pair

### Code for n-body integrations

I have started on the code for the N-body integrations in (nbody_sketch.py). The first part filters the data.
Your job will be to iterate through this dataset and perform N-body integrations for each exchange (index).
Good luck!!!





