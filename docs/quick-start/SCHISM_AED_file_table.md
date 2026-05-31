# SCHISM-AED file table

![SCHISM-AED file map](SCHISM_AED_filemap.png)

Quick reference for the files involved in a SCHISM-AED run. Sub-headings
within each table mirror the box names in the diagram above.

---

## Executables

<table class="bigtable">
  <thead>
    <tr><th style="width:24%">File</th><th style="width:38%">Description</th><th style="width:38%">Notes</th></tr>
  </thead>
  <tbody>
    <tr><td><code>pschism_AED</code></td>
        <td>Main SCHISM-AED MPI executable. Reads <code>param.nml</code> + <code>aed.nml</code>, advances the model, writes hydro + AED outputs.</td>
        <td>Launched with <code>mpirun -np N pschism_AED &lt;NSCRIBE&gt;</code>. NSCRIBE = ranks reserved for scribe-I/O; compute ranks = N − NSCRIBE.</td></tr>
    <tr><td><code>combine_output11_MPI</code></td>
        <td>Post-process binary that assembles per-rank AED output files into a single global-mesh file.</td>
        <td>Runs <strong>after</strong> <code>pschism_AED</code>. Usage: <code>mpirun -np 4 combine_output11_MPI -b &lt;begin&gt; -e &lt;end&gt; -o aed_data</code>. Reads <code>local_to_global_&lt;rank&gt;</code> maps.</td></tr>
  </tbody>
</table>

---

## Inputs

<table class="bigtable">
  <thead>
    <tr><th style="width:24%">File</th><th style="width:38%">Description</th><th style="width:38%">Notes</th></tr>
  </thead>
  <tbody>

    <tr class="section"><td colspan="3">Configuration <span class="hint">— purple boxes above the model</span></td></tr>

    <tr><td><code>param.nml</code></td>
        <td>SCHISM runtime configuration namelist.</td>
        <td>Time step, run length, output flags (<code>iof_hydro</code>), solver options, flux options (<code>iflux</code>), I/O cadence.</td></tr>
    <tr><td><code>aed.nml</code></td>
        <td>AED biogeochemistry configuration namelist.</td>
        <td>Active AED modules, state variables, diagnostics to save (<code>d_vars_save</code>), output compression level.</td></tr>
    <tr><td><code>aed/aed_macrophyte_pars.csv</code><br><code>aed/aed_phyto_pars.csv</code></td>
        <td><strong>BGC parameters (*.csv)</strong> — per-group parameter tables for the macrophyte and phytoplankton modules.</td>
        <td>Read by <code>aed.nml</code> when those modules are active. One row per group/strain.</td></tr>

    <tr class="section pagebreak"><td colspan="3">STATIC / SPATIAL <span class="hint">— read once at run start</span></td></tr>
    <tr class="subsection"><td colspan="3">Mesh &amp; vertical grid</td></tr>

    <tr><td><code>hgrid.gr3</code></td>
        <td>Horizontal unstructured mesh: node coordinates (projected/metric), depths, element connectivity.</td>
        <td>The model's geometric backbone — every other input/output is indexed off this mesh.</td></tr>
    <tr><td><code>hgrid.ll</code></td>
        <td>Same mesh in <strong>longitude/latitude</strong> coordinates.</td>
        <td>Used by sflux interpolation and any spherical-coordinate calcs.</td></tr>
    <tr><td><code>vgrid.in</code></td>
        <td>Vertical coordinate configuration.</td>
        <td>Defines sigma layers / LSC2 / hybrid scheme. For a 2D run typically 2 layers.</td></tr>

    <tr class="subsection"><td colspan="3">Spatial coefficients &amp; flags</td></tr>

    <tr><td><code>albedo.gr3</code></td><td>Surface albedo, per node.</td><td>Used in the radiation balance for heat-flux calcs.</td></tr>
    <tr><td><code>drag.gr3</code></td><td>Bottom-drag coefficient, per node.</td><td>Spatially-varying bottom friction.</td></tr>
    <tr><td><code>diffmax.gr3</code> / <code>diffmin.gr3</code></td><td>Maximum / minimum horizontal diffusivity, per node.</td><td>Cap and floor for <code>Dx</code>/<code>Dy</code> in the diffusion step.</td></tr>
    <tr><td><code>watertype.gr3</code></td><td>Jerlov water type / light-extinction class, per node.</td><td>Controls light penetration in the radiation model.</td></tr>
    <tr><td><code>windrot_geo2proj.gr3</code></td><td>Per-node rotation to convert geographic wind to projected-coords wind.</td><td>Bridges sflux (geographic) and model (projected) coordinate systems.</td></tr>
    <tr><td><code>fluxflag.prop</code></td><td>Per-element flag identifying flux-transect regions (−1..N).</td><td>Required for <code>iflux=2</code> boundary-flux accounting → contents of <code>flux.out</code>.</td></tr>
    <tr><td><code>tvd.prop</code></td><td>Per-element TVD on/off flag.</td><td>Selects which elements use the Total-Variation-Diminishing transport scheme.</td></tr>

    <tr class="subsection"><td colspan="3">Initial conditions (*.ic)</td></tr>

    <tr><td><code>elev.ic</code></td><td>Initial water-level field.</td><td>Per-node initial elevation.</td></tr>
    <tr><td><code>salt.ic</code></td><td>Initial salinity field.</td><td>Per-node initial salinity (PSU).</td></tr>
    <tr><td><code>temp.ic</code></td><td>Initial temperature field.</td><td>Per-node initial temperature (°C).</td></tr>

    <tr class="section pagebreak"><td colspan="3">TIME-VARYING <span class="hint">— read continually during the run</span></td></tr>
    <tr class="subsection"><td colspan="3">Ocean boundary (*.th, bctides)</td></tr>

    <tr><td><code>bctides.in</code></td><td>Tidal boundary configuration: nodes, constituents, amplitudes, phases.</td><td>Defines which open boundaries are tidal and how they're driven.</td></tr>
    <tr><td><code>elev.th</code></td><td>Time series of elevation at open boundaries.</td><td>Time column = seconds since <code>start_year/month/day</code>.</td></tr>
    <tr><td><code>TEM_1.th</code></td><td>Time series of temperature at boundary segment #1.</td><td>One file per active TEM boundary segment.</td></tr>
    <tr><td><code>AED_1.th</code> … <code>AED_17.th</code></td><td>Time series of <strong>AED tracer</strong> concentrations at each boundary segment.</td><td>One file per AED-tracer boundary segment (here 17 segments). Order matches <code>aed_models</code> in <code>aed.nml</code>.</td></tr>

    <tr class="subsection"><td colspan="3">Atmospheric forcing</td></tr>

    <tr><td><code>sflux/sflux_air_*.nc</code></td><td>NetCDF atmospheric forcing — wind components, air temperature, specific humidity.</td><td>One file per period (typically daily or stack-sized).</td></tr>
    <tr><td><code>sflux/sflux_rad_*.nc</code></td><td>NetCDF atmospheric forcing — shortwave / downward longwave radiation.</td><td>Same period structure as sflux_air.</td></tr>
    <tr><td><code>sflux/sflux_prc_*.nc</code></td><td>NetCDF atmospheric forcing — precipitation.</td><td>Optional — present only if precipitation is needed.</td></tr>

    <tr class="subsection"><td colspan="3">Sources &amp; sinks (rivers)</td></tr>

    <tr><td><code>source_sink.in</code></td><td>List of mesh <strong>element IDs</strong> where rivers / point sources enter.</td><td>This run has 14 sources; source #3 is Pioneer River main gauge. See <code>FLOOD.md</code> for the full mapping.</td></tr>
    <tr><td><code>vsource.th</code></td><td>Hourly <strong>flow time series</strong> (m³/s) for each source element.</td><td>Column 1 = time (seconds), columns 2…N+1 = flow at each source element.</td></tr>
    <tr><td><code>msource.th</code></td><td>Hourly <strong>concentration time series</strong> for each source × tracer combination.</td><td>Larger than vsource (one column per source × tracer).</td></tr>

  </tbody>
</table>

---

## Outputs

<table class="bigtable">
  <thead>
    <tr><th style="width:24%">File</th><th style="width:38%">Description</th><th style="width:38%">Notes</th></tr>
  </thead>
  <tbody>

    <tr class="section"><td colspan="3">ASCII outputs <span class="hint">— text logs and diagnostic dumps</span></td></tr>
    <tr class="subsection"><td colspan="3">Simulation logs</td></tr>

    <tr><td><code>mirror.out</code></td><td>Main runtime log — step counts, dt, mass-conservation diagnostics, init/finalize.</td><td>Rank 0 only. First place to check after a run.</td></tr>
    <tr><td><code>mirror.out.scribe</code></td><td>Companion log from the dedicated scribe-I/O rank.</td><td>Contains the full scribed-variables table (<code>iout_23d</code> per variable).</td></tr>
    <tr><td><code>fatal.error</code></td><td>Final abort message if <code>parallel_abort</code> was called.</td><td><strong>0 bytes = no fatal abort</strong>. Single rank-0 file.</td></tr>
    <tr><td><code>nonfatal_&lt;rank&gt;</code></td><td>Per-rank non-fatal warnings (e.g. Kriging matrix info).</td><td>One file per MPI rank. <code>grep</code>-able en masse.</td></tr>

    <tr class="subsection"><td colspan="3">Flux and diagnostic outputs</td></tr>

    <tr><td><code>flux.out</code></td><td>Per-region tracer transports across boundaries defined in <code>fluxflag.prop</code>.</td><td>Active when <code>iflux=1</code> or <code>iflux=2</code>. Drives boundary-flux mass balance (see <code>schism_run_process_3.ipynb</code> cell 42+).</td></tr>
    <tr><td><code>total.out</code></td><td>Domain-integrated hydrodynamic totals per output step (volume, kinetic energy).</td><td>Rank 0 only.</td></tr>
    <tr><td><code>total_TR.out</code></td><td>Same as <code>total.out</code> but for tracer state variables.</td><td>Cheapest sanity check for AED tracer mass balance.</td></tr>
    <tr><td><code>JCG.out</code></td><td>Pressure-solver convergence log: iterations + residual norms per step.</td><td>Diagnoses solver stalls / bad time steps.</td></tr>
    <tr><td><code>coriolis.out</code></td><td>One-off per-node dump of (lon, lat, Coriolis f).</td><td>Verification of the f-parameter field; no time dimension.</td></tr>
    <tr><td><code>subcycling.out</code></td><td>Per-step count of barotropic subcycles needed (1 = none).</td><td>Stability monitor — max value flags the most stressed step.</td></tr>
    <tr><td><code>maxelev_&lt;rank&gt;</code></td><td>Per-rank running maximum elevation at each node.</td><td>Inundation envelope — useful for storm-surge analysis.</td></tr>
    <tr><td><code>maxdahv_&lt;rank&gt;</code></td><td>Per-rank running maximum depth-averaged horizontal velocity at each node.</td><td>Useful for scour / sediment-transport analysis.</td></tr>
    <tr><td><code>local_to_global_&lt;rank&gt;</code><br><code>global_to_local.prop</code></td><td>Per-rank partition map (local → global node/element IDs) and its inverse.</td><td>Required by <code>combine_output11_MPI</code>. Not data per se but indispensable.</td></tr>

    <tr class="section"><td colspan="3">NetCDF outputs <span class="hint">— binary, per-stack, optionally compressed</span></td></tr>
    <tr class="subsection"><td colspan="3">Hotstart / restart</td></tr>

    <tr><td><code>hotstart_&lt;rank&gt;_&lt;stack&gt;.nc</code></td><td>Per-rank snapshots of the full model state (hydro + AED).</td><td>Needed to continue a run from a checkpoint. Written at intervals set by <code>nhot_write</code> in <code>param.nml</code>. Combine with <code>combine_hotstart7</code> before reuse.</td></tr>

    <tr class="subsection"><td colspan="3">Scribed (one NC per variable, per stack)</td></tr>

    <tr><td><code>out2d_&lt;S&gt;.nc</code></td><td>2D fields aggregated into a single NC per stack: dry-cell flags, elevation, bottom stress, depth-averaged velocity, plus a couple of element-centred AED state vars.</td><td>Always written. Stack <code>&lt;S&gt;</code> is the time-chunk number.</td></tr>
    <tr><td><code>salinity_&lt;S&gt;.nc</code></td><td>Salinity (3D, node-centred).</td><td><code>iof_hydro(19)=1</code>.</td></tr>
    <tr><td><code>temperature_&lt;S&gt;.nc</code></td><td>Water temperature (3D, node-centred).</td><td>Requires <code>iof_hydro(18)=1</code>.</td></tr>
    <tr><td><code>horizontalVelX_&lt;S&gt;.nc</code><br><code>horizontalVelY_&lt;S&gt;.nc</code></td><td>X / Y components of horizontal velocity (3D, node-centred).</td><td><code>iof_hydro(16)=1</code>.</td></tr>
    <tr><td><code>zCoordinates_&lt;S&gt;.nc</code></td><td>Z-coordinate of each layer/node (3D).</td><td>Needed for any vertical-profile analysis.</td></tr>
    <tr><td><code>OXY_oxy_&lt;S&gt;.nc</code>, <code>NIT_amm_&lt;S&gt;.nc</code>, <code>NIT_nit_&lt;S&gt;.nc</code>, <code>OGM_*_&lt;S&gt;.nc</code>, <code>PHS_frp_&lt;S&gt;.nc</code>, …</td><td>One file per AED state variable, 3D node-centred.</td><td>Scribed automatically whenever AED is on. Filename = variable name.</td></tr>

    <tr class="subsection"><td colspan="3">Unscribed (per-rank AED files)</td></tr>

    <tr><td><code>aed_data_&lt;rank&gt;_&lt;stack&gt;.nc</code></td><td>Per-MPI-rank, per-stack NetCDF with AED diagnostic variables (the <code>d_vars_save</code> list from <code>aed.nml</code>).</td><td>Written directly by <code>pschism_AED</code>; element/face-centred. Must be assembled via <code>combine_output11_MPI</code> before analysis. With deflate enabled (<code>nc_deflate_level=4</code>), each file is ~10–15 MB.</td></tr>

    <tr class="subsection"><td colspan="3">Combined (after combine_output11_MPI)</td></tr>

    <tr><td><code>aed_data_cmb_&lt;stack&gt;.nc</code></td><td>Global-mesh assembly of all per-rank AED files for one stack.</td><td>What every analysis / animation script reads. Contains both 3D pelagic diagnostics (e.g. <code>OXY_sat</code>, <code>NIT_nitrif</code>) and 2D sheet diagnostics (e.g. <code>OXY_oxy_dsf</code>, <code>OGM_pon_swi</code>). Stack-1 file is typically 400–500 MB after compression.</td></tr>

  </tbody>
</table>

## Quick start guide {.pagebreak}

### 1. Building from source

On a fresh Ubuntu host, the full SCHISM-AED build flow is:

```bash
git clone https://github.com/AquaticEcoDynamics/AED_Tools
cd AED_Tools/
./fetch_sources.sh schism            # pull the SCHISM source tree
./fetch_sources.sh all               # pull the AED libraries and helpers
sudo apt install libopenmpi-dev
sudo apt install libnetcdf-dev libnetcdff-dev
ln -s schism a                       # convenience symlink for the patch
patch -p0 < schism-aed/aed-schism.xdiff
cd schism/src
ln -s ../../schism-aed/src/AED .     # graft AED into SCHISM's source tree
cd ../..
./build_schism.sh --with-aed         # compile pschism_AED + combine_output11_MPI
ls binaries/ubuntu/22.04/            # confirm the executables landed here
```

<div class="two-col" markdown="1">
<div markdown="1">

### 2. Running the model

Place `param.nml`, `aed.nml`, all `*.gr3` / `*.ic` / `*.th` / `bctides.in`
inputs and the `sflux/` forcing directory in a single run directory.
Then either invoke `mpirun` directly:

```bash
# In the run directory:
mpirun -np 64 pschism_AED 24
# 64 ranks total; last integer is NSCRIBE
# (ranks reserved for scribe-I/O).
# Compute ranks = NPROC - NSCRIBE = 40.
```

…or use the wrapper script which also runs
the post-process combine step in one go:

```bash
./run_schism_vm.sh
# Override defaults via env vars, e.g.:
NPROC=32 NSCRIBE=8 COMBINE_END=14 ./run_schism_vm.sh
```

</div>
<div markdown="1">

### 3. At-a-glance run health check

After a run completes, the cheap sanity-check sequence is:

```bash
cd outputs
cat fatal.error
   # should be empty (0 bytes = no fatal abort)
tail mirror.out
   # should end with a clean finalization message
grep -i error nonfatal_*
   # quick scan for rank-local warnings
ls aed_data_cmb_*.nc
   # confirm combined AED files exist
```

If those look right, the model ran cleanly and
the data is ready to analyse.

</div>
</div>
