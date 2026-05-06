/* ═══════════════════════════════════════════════════════════════
   Lattice Boltzmann Wind Tunnel — D2Q9 BGK
   ═══════════════════════════════════════════════════════════════ */

(() => {
  'use strict';

  /* ── Grid & State ──────────────────────────────────────── */
  let xdim = 400, ydim = 200;

  // Distribution arrays — one per lattice direction (SoA layout)
  // Naming: the "direction the particles are going"
  //   nE = particles heading east, etc.
  // Screen coords: x→right, y→down. So "N" = y-1, "S" = y+1.
  let n0, nE, nW, nN, nS, nNE, nNW, nSE, nSW;
  let bar;       // obstacle mask
  let ux, uy, rh; // macroscopic fields
  let curlField, spdField;

  /* ── Parameters ────────────────────────────────────────── */
  let reynolds = 180;
  let u0 = 0.1;            // inlet speed, lattice units (< ~0.15 for stability)
  let omega;                // 1/tau
  let stepsPerFrame = 20;
  let paused = false;
  let vizMode = 'vorticity';
  let drawMode = 'move';
  let currentPreset = 'cylinder';
  let stepCount = 0;

  /* ── Streamline particles ──────────────────────────────── */
  const NPART = 2000;
  let parts = [];
  let trailCv, trailCx;

  /* ── Canvas ────────────────────────────────────────────── */
  const canvas = document.getElementById('sim-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  let imgData, imgBuf;

  /* ── Physics helpers ───────────────────────────────────── */
  function updateOmega() {
    const L = ydim / 3.5; // characteristic length ≈ obstacle diameter
    const nu = u0 * L / reynolds;
    const tau = 3 * nu + 0.5;
    omega = 1 / Math.max(tau, 0.505);
  }

  /* ── Memory ────────────────────────────────────────────── */
  function allocArrays() {
    const n = xdim * ydim;
    n0  = new Float32Array(n); nE  = new Float32Array(n); nW  = new Float32Array(n);
    nN  = new Float32Array(n); nS  = new Float32Array(n);
    nNE = new Float32Array(n); nNW = new Float32Array(n);
    nSE = new Float32Array(n); nSW = new Float32Array(n);
    bar  = new Uint8Array(n);
    ux  = new Float32Array(n); uy  = new Float32Array(n); rh  = new Float32Array(n);
    curlField = new Float32Array(n); spdField = new Float32Array(n);
  }

  /* ── Set a cell to equilibrium ─────────────────────────── */
  function setEq(i, vx, vy, rho0) {
    const ux3  = 3 * vx;
    const uy3  = 3 * vy;
    const ux2  = vx * vx;
    const uy2  = vy * vy;
    const uxuy = vx * vy;
    const u2   = ux2 + uy2;
    const u215 = 1.5 * u2;
    n0[i]  = (4/9)  * rho0 * (1                                    - u215);
    nE[i]  = (1/9)  * rho0 * (1 + ux3            + 4.5 * ux2       - u215);
    nW[i]  = (1/9)  * rho0 * (1 - ux3            + 4.5 * ux2       - u215);
    nN[i]  = (1/9)  * rho0 * (1 - uy3            + 4.5 * uy2       - u215);
    nS[i]  = (1/9)  * rho0 * (1 + uy3            + 4.5 * uy2       - u215);
    nNE[i] = (1/36) * rho0 * (1 + ux3 - uy3 + 4.5*(u2 - 2*uxuy)   - u215);
    nNW[i] = (1/36) * rho0 * (1 - ux3 - uy3 + 4.5*(u2 + 2*uxuy)   - u215);
    nSE[i] = (1/36) * rho0 * (1 + ux3 + uy3 + 4.5*(u2 + 2*uxuy)   - u215);
    nSW[i] = (1/36) * rho0 * (1 - ux3 + uy3 + 4.5*(u2 - 2*uxuy)   - u215);
    rh[i] = rho0; ux[i] = vx; uy[i] = vy;
  }

  /*
    Convention for the 9 directions in screen coordinates (x→right, y→down):
      0: rest    (0, 0)
      E: east    (+1, 0)
      W: west    (-1, 0)
      N: north   (0, -1)   — up on screen
      S: south   (0, +1)   — down on screen
      NE: ne     (+1, -1)
      NW: nw     (-1, -1)
      SE: se     (+1, +1)
      SW: sw     (-1, +1)

    Macroscopic velocity:
      rho = sum(fi)
      ux  = (nE + nNE + nSE - nW - nNW - nSW) / rho
      uy  = (nS + nSE + nSW - nN - nNE - nNW) / rho   (positive = downward)

    Equilibrium: feq_i = w_i * rho * (1 + 3*(ci·u) + 4.5*(ci·u)² - 1.5*|u|²)
      where ci·u = cx_i*ux + cy_i*uy  with  cx/cy from the table above.

    For nN  (cx=0, cy=-1): ci·u = -uy  →  3*(-uy) = -uy3;  4.5*(-uy)² = 4.5*uy²
    For nNE (cx=1, cy=-1): ci·u = ux-uy → 3*(ux-uy); 4.5*(ux-uy)² = 4.5*(ux²+uy²-2*ux*uy)
  */

  /* ── Initialize flow ───────────────────────────────────── */
  function initFlow() {
    stepCount = 0;
    for (let y = 0; y < ydim; y++) {
      for (let x = 0; x < xdim; x++) {
        const i = y * xdim + x;
        if (bar[i]) {
          n0[i]=nE[i]=nW[i]=nN[i]=nS[i]=nNE[i]=nNW[i]=nSE[i]=nSW[i]=0;
          rh[i]=ux[i]=uy[i]=0;
          continue;
        }
        // Perturbation to seed vortex shedding
        const vyPert = (x > xdim*0.1 && x < xdim*0.4)
          ? 0.03 * Math.sin(2*Math.PI*y/ydim*6)
          : 0;
        setEq(i, u0, vyPert, 1);
      }
    }
  }

  /* ── Obstacle presets ──────────────────────────────────── */
  function clearBar() { bar.fill(0); }

  function placePreset(name) {
    clearBar();
    const ox = Math.floor(xdim * 0.22);
    const oy = Math.floor(ydim / 2);
    const R  = Math.floor(ydim / 7);

    switch (name) {
      case 'cylinder':
        for (let y = 0; y < ydim; y++)
          for (let x = 0; x < xdim; x++) {
            if ((x-ox)*(x-ox)+(y-oy)*(y-oy) <= R*R) bar[y*xdim+x]=1;
          }
        break;
      case 'airfoil': {
        const ch = Math.floor(ydim/3.5), th = 0.15, sx = ox-Math.floor(ch/2);
        for (let i = 0; i <= ch; i++) {
          const t = i/ch;
          const ht = ch*th*(2.969*Math.sqrt(t)-1.26*t-3.516*t*t+2.843*t*t*t-1.015*t*t*t*t);
          for (let y = Math.floor(oy-ht); y <= Math.ceil(oy+ht); y++) {
            const px = sx+i;
            if (px>=0&&px<xdim&&y>=0&&y<ydim) bar[y*xdim+px]=1;
          }
        }
        break;
      }
      case 'bridge': {
        const hw = Math.floor(ydim/16), hh = Math.floor(ydim/4);
        for (let y = oy-hh; y <= oy+hh; y++) {
          const fr = (y-(oy-hh))/(2*hh);
          const w = Math.floor(hw*(0.6+0.4*fr));
          for (let x = ox-w; x <= ox+w; x++)
            if (x>=0&&x<xdim&&y>=0&&y<ydim) bar[y*xdim+x]=1;
        }
        break;
      }
      case 'plate': {
        const hh = Math.floor(ydim/6);
        for (let y = oy-hh; y <= oy+hh; y++)
          if (y>=0&&y<ydim) { bar[y*xdim+ox]=1; if(ox+1<xdim) bar[y*xdim+ox+1]=1; }
        break;
      }
    }
  }

  /* ── LBM step ──────────────────────────────────────────── */
  function simulate() {
    const xs = xdim, ys = ydim;

    // === Collide ===
    for (let y = 0; y < ys; y++) {
      for (let x = 0; x < xs; x++) {
        const i = y*xs + x;
        if (bar[i]) continue;

        let r = n0[i]+nE[i]+nW[i]+nN[i]+nS[i]+nNE[i]+nNW[i]+nSE[i]+nSW[i];
        if (r < 0.01) r = 0.01;  // prevent division by near-zero density
        let vx = (nE[i]+nNE[i]+nSE[i] - nW[i]-nNW[i]-nSW[i]) / r;
        let vy = (nS[i]+nSE[i]+nSW[i] - nN[i]-nNE[i]-nNW[i]) / r;

        // Clamp velocity to prevent numerical explosion
        // LBM is stable when |v| << 1/sqrt(3) ≈ 0.577; clamp well below
        const speed2 = vx*vx + vy*vy;
        if (speed2 > 0.04) {  // |v| > 0.2
          const scale = 0.2 / Math.sqrt(speed2);
          vx *= scale;
          vy *= scale;
        }

        rh[i] = r; ux[i] = vx; uy[i] = vy;

        const ux3=3*vx, uy3=3*vy;
        const ux2=vx*vx, uy2=vy*vy, uxuy2=2*vx*vy;
        const u2=ux2+uy2, u215=1.5*u2;

        n0[i]  += omega*((4/9) *r*(1                              -u215) - n0[i]);
        nE[i]  += omega*((1/9) *r*(1 + ux3            + 4.5*ux2  -u215) - nE[i]);
        nW[i]  += omega*((1/9) *r*(1 - ux3            + 4.5*ux2  -u215) - nW[i]);
        nN[i]  += omega*((1/9) *r*(1 - uy3            + 4.5*uy2  -u215) - nN[i]);
        nS[i]  += omega*((1/9) *r*(1 + uy3            + 4.5*uy2  -u215) - nS[i]);
        nNE[i] += omega*((1/36)*r*(1 + ux3-uy3 + 4.5*(u2-uxuy2) -u215) - nNE[i]);
        nNW[i] += omega*((1/36)*r*(1 - ux3-uy3 + 4.5*(u2+uxuy2) -u215) - nNW[i]);
        nSE[i] += omega*((1/36)*r*(1 + ux3+uy3 + 4.5*(u2+uxuy2) -u215) - nSE[i]);
        nSW[i] += omega*((1/36)*r*(1 - ux3+uy3 + 4.5*(u2-uxuy2) -u215) - nSW[i]);
      }
    }

    // === Stream (in-place, careful scan order) ===
    // nN  moves to y-1: scan y from 0 to ys-2 (top→bottom, pulling from below)
    for (let y = 0; y < ys-1; y++)
      for (let x = 0; x < xs; x++)
        nN[y*xs+x] = nN[(y+1)*xs+x];

    // nNE moves to (x+1, y-1): scan y top→bottom, x right→left
    for (let y = 0; y < ys-1; y++)
      for (let x = xs-1; x > 0; x--)
        nNE[y*xs+x] = nNE[(y+1)*xs+(x-1)];

    // nNW moves to (x-1, y-1): scan y top→bottom, x left→right
    for (let y = 0; y < ys-1; y++)
      for (let x = 0; x < xs-1; x++)
        nNW[y*xs+x] = nNW[(y+1)*xs+(x+1)];

    // nS  moves to y+1: scan y from ys-1 to 1 (bottom→top, pulling from above)
    for (let y = ys-1; y > 0; y--)
      for (let x = 0; x < xs; x++)
        nS[y*xs+x] = nS[(y-1)*xs+x];

    // nSE moves to (x+1, y+1): scan y bottom→top, x right→left
    for (let y = ys-1; y > 0; y--)
      for (let x = xs-1; x > 0; x--)
        nSE[y*xs+x] = nSE[(y-1)*xs+(x-1)];

    // nSW moves to (x-1, y+1): scan y bottom→top, x left→right
    for (let y = ys-1; y > 0; y--)
      for (let x = 0; x < xs-1; x++)
        nSW[y*xs+x] = nSW[(y-1)*xs+(x+1)];

    // nE  moves to x+1: scan x from xs-1 to 1 (right→left)
    for (let y = 0; y < ys; y++)
      for (let x = xs-1; x > 0; x--)
        nE[y*xs+x] = nE[y*xs+(x-1)];

    // nW  moves to x-1: scan x from 0 to xs-2 (left→right)
    for (let y = 0; y < ys; y++)
      for (let x = 0; x < xs-1; x++)
        nW[y*xs+x] = nW[y*xs+(x+1)];

    // === Bounce-back at obstacles ===
    for (let y = 1; y < ys-1; y++) {
      for (let x = 1; x < xs-1; x++) {
        const i = y*xs+x;
        if (!bar[i]) continue;
        // Each direction that entered this obstacle cell gets reflected back out
        const iN=i-xs, iS=i+xs, iE=i+1, iW=i-1;
        const iNE=i-xs+1, iNW=i-xs-1, iSE=i+xs+1, iSW=i+xs-1;

        // A particle heading east (nE) that arrived here came from the west neighbor.
        // Bounce it back: the west neighbor gets nW (heading west) from this cell's nE.
        nW[iW]   = nE[i];
        nE[iE]   = nW[i];
        nS[iS]   = nN[i];
        nN[iN]   = nS[i];
        nSW[iSW] = nNE[i];
        nNW[iNW] = nSE[i];
        nSE[iSE] = nNW[i];
        nNE[iNE] = nSW[i];
      }
    }

    // === Inlet BC (x=0): equilibrium with small oscillation ===
    for (let y = 0; y < ys; y++) {
      const pert = 0.05 * u0 * Math.sin(2*Math.PI * stepCount / 200);
      setEq(y*xs, u0, pert, 1);
    }

    // === Outlet BC (x=xs-1): zero-gradient ===
    for (let y = 0; y < ys; y++) {
      const d = y*xs+xs-1, s = y*xs+xs-2;
      n0[d]=n0[s]; nE[d]=nE[s]; nW[d]=nW[s]; nN[d]=nN[s]; nS[d]=nS[s];
      nNE[d]=nNE[s]; nNW[d]=nNW[s]; nSE[d]=nSE[s]; nSW[d]=nSW[s];
    }

    // === Top/bottom: periodic (wrap around) ===
    for (let x = 0; x < xs; x++) {
      // Bottom row (y=ys-1) receives from top row (y=0) via streaming
      // But since streaming didn't wrap, copy manually:
      nS[x]            = nS[(ys-1)*xs+x];    // what fell off the top goes to bottom
      nN[(ys-1)*xs+x]  = nN[x];              // what fell off bottom goes to top
      nSE[x]           = nSE[(ys-1)*xs+x];
      nSW[x]           = nSW[(ys-1)*xs+x];
      nNE[(ys-1)*xs+x] = nNE[x];
      nNW[(ys-1)*xs+x] = nNW[x];
    }

    stepCount++;
  }

  /* ── Derived fields ────────────────────────────────────── */
  function computeDerived() {
    let mxS = 0, mxC = 0;
    for (let y = 1; y < ydim-1; y++) {
      for (let x = 1; x < xdim-1; x++) {
        const i = y*xdim+x;
        const s = Math.sqrt(ux[i]*ux[i]+uy[i]*uy[i]);
        spdField[i] = s;
        if (s > mxS) mxS = s;
        // Curl = duy/dx - dux/dy
        const c = (uy[i+1]-uy[i-1]) - (ux[i+xdim]-ux[i-xdim]);
        curlField[i] = c;
        const ac = Math.abs(c);
        if (ac > mxC) mxC = ac;
      }
    }
    for (let x = 0; x < xdim; x++) {
      spdField[x]=spdField[xdim+x]; curlField[x]=curlField[xdim+x];
      spdField[(ydim-1)*xdim+x]=spdField[(ydim-2)*xdim+x];
      curlField[(ydim-1)*xdim+x]=curlField[(ydim-2)*xdim+x];
    }
    for (let y = 0; y < ydim; y++) {
      spdField[y*xdim]=spdField[y*xdim+1]; curlField[y*xdim]=curlField[y*xdim+1];
      spdField[y*xdim+xdim-1]=spdField[y*xdim+xdim-2];
      curlField[y*xdim+xdim-1]=curlField[y*xdim+xdim-2];
    }
    return { mxS, mxC };
  }

  /* ── Colormaps ─────────────────────────────────────────── */
  function velRGB(t) {
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.25) {
      const s=t*4; r=68+s*(49-68)|0; g=1+s*(104-1)|0; b=84+s*(172-84)|0;
    } else if (t < 0.5) {
      const s=(t-.25)*4; r=49+s*(53-49)|0; g=104+s*(183-104)|0; b=172+s*(121-172)|0;
    } else if (t < 0.75) {
      const s=(t-.5)*4; r=53+s*(187-53)|0; g=183+s*(210-183)|0; b=121+s*(35-121)|0;
    } else {
      const s=(t-.75)*4; r=187+s*(253-187)|0; g=210+s*(231-210)|0; b=35+s*(37-35)|0;
    }
    return [r, g, b];
  }

  function curlRGB(t) {
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.5) {
      const s=t*2; r=10+s*230|0; g=50+s*190|0; b=220+s*25|0;
    } else {
      const s=(t-.5)*2; r=240-s*20|0; g=240-s*192|0; b=245-s*207|0;
    }
    return [r, g, b];
  }

  // LUTs in ABGR uint32 format
  const LUT = 512;
  const velLUT = new Uint32Array(LUT);
  const curlLUT = new Uint32Array(LUT);
  for (let i = 0; i < LUT; i++) {
    const t = i/(LUT-1);
    const v = velRGB(t);
    velLUT[i] = 0xFF000000 | (v[2]<<16) | (v[1]<<8) | v[0];
    const c = curlRGB(t);
    curlLUT[i] = 0xFF000000 | (c[2]<<16) | (c[1]<<8) | c[0];
  }
  const OBS_ABGR = 0xFF20201E; // rgb(30,32,32) — matches --bg-panel

  /* ── Render ────────────────────────────────────────────── */
  function render() {
    const { mxS, mxC } = computeDerived();

    if (vizMode === 'streamlines') {
      renderStreamlines(mxS);
      updateColorbar('streamlines');
      return;
    }

    const px = new Uint32Array(imgBuf.buffer);
    const n = xdim * ydim;
    const isV = vizMode === 'vorticity';
    const lut = isV ? curlLUT : velLUT;

    // Contrast: for vorticity, cap at 15% of max so the wake really pops
    const cap = isV ? Math.max(mxC * 0.15, 1e-6) : Math.max(mxS, 1e-6);

    for (let i = 0; i < n; i++) {
      if (bar[i]) { px[i] = OBS_ABGR; continue; }
      let ci;
      if (isV) {
        const norm = curlField[i] / cap;           // range ~ [-6.67, 6.67]
        const t = 0.5 + 0.5 * Math.max(-1, Math.min(1, norm)); // clamp to [0,1]
        ci = (t * (LUT-1)) | 0;
      } else {
        ci = (spdField[i] / cap * (LUT-1)) | 0;
        if (ci >= LUT) ci = LUT - 1;
      }
      px[i] = lut[ci];
    }
    ctx.putImageData(imgData, 0, 0);
    updateColorbar(vizMode);
  }

  /* ── Streamlines ───────────────────────────────────────── */
  function initParticles() {
    parts = [];
    for (let i = 0; i < NPART; i++) parts.push(spawnP());
    if (!trailCv) { trailCv = document.createElement('canvas'); trailCx = trailCv.getContext('2d'); }
    trailCv.width = xdim; trailCv.height = ydim;
    trailCx.fillStyle = '#18181a'; trailCx.fillRect(0,0,xdim,ydim);
  }
  function spawnP() { return {x:Math.random()*xdim*0.02, y:Math.random()*ydim, age:Math.random()*100|0, max:600+(Math.random()*400|0)}; }

  function renderStreamlines(mxS) {
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      const ix=p.x|0, iy=p.y|0;
      if (ix<0||ix>=xdim||iy<0||iy>=ydim||p.age>=p.max) { parts[i]=spawnP(); continue; }
      const idx=iy*xdim+ix;
      if (bar[idx]) { parts[i]=spawnP(); continue; }
      p.x += ux[idx]*3.0; p.y += uy[idx]*3.0; p.age++;
    }
    trailCx.fillStyle = 'rgba(24,24,26,0.035)';
    trailCx.fillRect(0,0,xdim,ydim);
    const inv = mxS>1e-6 ? 1/mxS : 1;
    for (const p of parts) {
      const ix=p.x|0, iy=p.y|0;
      if (ix<0||ix>=xdim||iy<0||iy>=ydim) continue;
      const t = Math.min(1, spdField[iy*xdim+ix]*inv);
      const [r,g,b] = velRGB(t);
      trailCx.fillStyle = `rgba(${r},${g},${b},${Math.max(0.1,1-p.age/p.max)})`;
      trailCx.fillRect(p.x,p.y,1.5,1.5);
    }
    // Obstacles on top
    const td = trailCx.getImageData(0,0,xdim,ydim), tp = td.data;
    for (let i = 0; i < xdim*ydim; i++) if (bar[i]) { const p=i*4; tp[p]=30;tp[p+1]=32;tp[p+2]=32;tp[p+3]=255; }
    trailCx.putImageData(td,0,0);
    ctx.drawImage(trailCv,0,0);
  }

  /* ── Colorbar ──────────────────────────────────────────── */
  function updateColorbar(mode) {
    const cb = document.getElementById('colorbar-canvas');
    const cc = cb.getContext('2d');
    const w=cb.width, h=cb.height, img=cc.createImageData(w,h), d=img.data;
    for (let x = 0; x < w; x++) {
      const t = x/(w-1);
      const [r,g,b] = mode==='vorticity' ? curlRGB(t) : velRGB(t);
      for (let y = 0; y < h; y++) { const p=(y*w+x)*4; d[p]=r; d[p+1]=g; d[p+2]=b; d[p+3]=255; }
    }
    cc.putImageData(img,0,0);
    const lbl=document.getElementById('colorbar-label'), mn=document.getElementById('colorbar-min'), mx=document.getElementById('colorbar-max');
    if (mode==='vorticity') { lbl.textContent='Vorticity (curl)'; mn.textContent='CW'; mx.textContent='CCW'; }
    else if (mode==='velocity') { lbl.textContent='Velocity magnitude'; mn.textContent='0'; mx.textContent='max'; }
    else { lbl.textContent='Streamlines \u2014 colored by speed'; mn.textContent='slow'; mx.textContent='fast'; }
  }

  /* ── Educational content ───────────────────────────────── */
  const regimeText = {
    laminar: { title:'Laminar Flow', text:`At Re\u00a0=\u00a0<strong id="re-inline">{RE}</strong>, viscous forces dominate. The fluid wraps smoothly around the obstacle with no separation \u2014 a perfectly steady, symmetric flow. This is how honey flows around a spoon, or how blood moves through your smallest capillaries.<br><br>In this regime, the flow is entirely predictable. If you could reverse time, the fluid would trace the exact same path backward. There is no chaos here.` },
    vortexStreet: { title:'Von K\u00e1rm\u00e1n Vortex Street', text:`Re\u00a0=\u00a0<strong id="re-inline">{RE}</strong> \u2014 you\u2019re watching one of fluid dynamics\u2019 most beautiful phenomena. The flow separates behind the obstacle and forms alternating vortices that peel off in a staggered pattern \u2014 a <strong>von K\u00e1rm\u00e1n vortex street</strong>, first described mathematically by Theodore von K\u00e1rm\u00e1n in 1911 (\u201c\u00dcber den Mechanismus des Widerstandes\u201d, G\u00f6ttinger Nachrichten).<br><br>This is what makes power lines hum on windy days. It\u2019s what caused the <strong>Tacoma Narrows Bridge</strong> to oscillate wildly and collapse on November 7, 1940 \u2014 the vortex shedding frequency matched the bridge\u2019s natural frequency, creating resonance.<br><br>The shedding frequency follows the <strong>Strouhal number</strong> St\u00a0\u2248\u00a00.2 (Roshko, 1954, NACA Report 1191), meaning the vortices peel off at a remarkably consistent rate.` },
    turbulentWake: { title:'Turbulent Wake', text:`At Re\u00a0=\u00a0<strong id="re-inline">{RE}</strong>, the orderly vortex street is breaking apart. The wake behind the obstacle becomes increasingly chaotic \u2014 vortices merge, split, and interact unpredictably.<br><br>This transition from order to chaos is one of the <strong>deepest unsolved problems in physics</strong>. We can describe it statistically but can\u2019t predict it from first principles. Werner Heisenberg reportedly said: \u201cWhen I meet God, I am going to ask him two questions: Why relativity? And why turbulence? I really believe he will have an answer for the first.\u201d<br><br>Yet this \u201cmessy\u201d turbulence is everywhere \u2014 it mixes cream into your coffee, keeps airplanes aloft, and drives weather patterns across the planet.` },
    turbulent: { title:'Fully Turbulent', text:`Re\u00a0=\u00a0<strong id="re-inline">{RE}</strong> \u2014 fully turbulent. The wake is chaotic across all scales. Small eddies cascade from large ones in what Kolmogorov described in 1941 (\u201cThe Local Structure of Turbulence\u201d, Dokl. Akad. Nauk SSSR) \u2014 the <strong>Kolmogorov cascade</strong>.<br><br>Understanding turbulence is one of the <strong>Clay Millennium Prize problems</strong> (claymath.org, 2000) \u2014 a million-dollar bounty for proving basic properties of the Navier-Stokes equations. After 200+ years of study, we still can\u2019t prove whether smooth solutions always exist. See Fefferman (2006), \u201cExistence and Smoothness of the Navier-Stokes Equation\u201d for the formal problem statement.<br><br>Every time you look at a waterfall, a campfire, or clouds forming, you\u2019re watching a phenomenon we can simulate but fundamentally cannot solve analytically.` }
  };
  const obsText = {
    cylinder: `The <strong>cylinder</strong> is the classic demonstration shape for vortex shedding. Its perfect symmetry makes the instability especially vivid \u2014 the symmetric wake becomes unstable and spontaneously breaks into the alternating pattern. This is the geometry Theodore von K\u00e1rm\u00e1n studied when he discovered these vortex streets in 1911.`,
    airfoil: `The <strong>airfoil</strong> is the shape of a wing. At low Reynolds numbers, flow stays attached and generates lift. At higher Re or steeper angles, the flow separates from the top surface \u2014 <strong>stall</strong>. The asymmetric shape means the vortex pattern is different from a cylinder: the wake tends to deflect, reflecting the lift force.`,
    bridge: `The <strong>bridge pylon</strong> demonstrates why engineers must account for vortex-induced vibration. The Tacoma Narrows Bridge (1940) collapsed because its deck profile created strong periodic vortex shedding that matched the structure\u2019s resonant frequency.`,
    plate: `The <strong>flat plate</strong> perpendicular to the flow is the highest-drag shape possible. Sharp edges force immediate flow separation, creating a wide turbulent wake. Drag coefficient ~2.0 \u2014 roughly 50\u00d7 that of a streamlined airfoil. This is why parachutes work.`
  };
  const vizText = {
    velocity: `<strong>Velocity magnitude</strong> shows how fast the fluid moves at each point. Dark = slow (stagnation behind obstacle), bright = fast (acceleration around edges). Viridis-like colormap: purple \u2192 green \u2192 yellow.`,
    vorticity: `<strong>Vorticity</strong> shows local rotation \u2014 the fluid\u2019s \u201ccurl.\u201d Blue = clockwise, red = counter-clockwise, white = none. Best mode for seeing vortex streets: each vortex core lights up as a distinct colored spot.`,
    streamlines: `<strong>Streamlines</strong> show paths that fluid particles follow. Each dot is a tracer being carried by the flow. Color = local speed. This is what you\u2019d see injecting dye into a real wind tunnel.`
  };

  function updateEdu() {
    let reg;
    if (reynolds < 40) reg='laminar';
    else if (reynolds < 200) reg='vortexStreet';
    else if (reynolds < 1000) reg='turbulentWake';
    else reg='turbulent';
    const c = regimeText[reg];
    document.getElementById('regime-title').textContent = c.title;
    document.getElementById('regime-text').innerHTML = c.text.replace('{RE}', reynolds);
    document.getElementById('obstacle-text').innerHTML = obsText[currentPreset]||'';
    document.getElementById('viz-text').innerHTML = vizText[vizMode]||'';
  }

  /* ── Interaction ────────────────────────────────────────── */
  let drawing = false;
  let draggingObstacle = false;
  let dragStartX = 0, dragStartY = 0;
  let obsCenterX = 0, obsCenterY = 0;

  function gxy(cx,cy) { const r=canvas.getBoundingClientRect(); return {x:((cx-r.left)/r.width*xdim)|0, y:((cy-r.top)/r.height*ydim)|0}; }

  function paint(gx,gy) {
    const r=Math.max(2,(ydim/50)|0), val=drawMode==='draw'?1:0;
    for (let dy=-r;dy<=r;dy++) for (let dx=-r;dx<=r;dx++) {
      if (dx*dx+dy*dy>r*r) continue;
      const x=gx+dx, y=gy+dy;
      if (x>1&&x<xdim-2&&y>1&&y<ydim-2) { const i=y*xdim+x; bar[i]=val; if(!val) setEq(i,u0,0,1); }
    }
  }

  // Check if a grid point is near an obstacle
  function nearObstacle(gx, gy, radius) {
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const x = gx+dx, y = gy+dy;
        if (x>=0 && x<xdim && y>=0 && y<ydim && bar[y*xdim+x]) return true;
      }
    }
    return false;
  }

  // Find center of mass of all obstacle cells
  function obsCenter() {
    let sx=0, sy=0, n=0;
    for (let y=0; y<ydim; y++) for (let x=0; x<xdim; x++) {
      if (bar[y*xdim+x]) { sx+=x; sy+=y; n++; }
    }
    return n>0 ? {x: sx/n, y: sy/n} : {x: xdim/2, y: ydim/2};
  }

  // Move all obstacle cells by dx, dy
  function moveObstacle(dx, dy) {
    const oldBar = new Uint8Array(bar);
    bar.fill(0);
    for (let y=0; y<ydim; y++) for (let x=0; x<xdim; x++) {
      if (oldBar[y*xdim+x]) {
        const nx = x+dx, ny = y+dy;
        if (nx>1 && nx<xdim-2 && ny>1 && ny<ydim-2) {
          bar[ny*xdim+nx] = 1;
        }
      }
    }
    // Re-equilibrate newly exposed fluid cells
    for (let i=0; i<xdim*ydim; i++) {
      if (oldBar[i] && !bar[i]) setEq(i, u0, 0, 1);
    }
  }

  canvas.addEventListener('pointerdown', e => {
    const g = gxy(e.clientX, e.clientY);
    e.preventDefault();

    if (drawMode === 'move') {
      // Check if clicking near obstacle
      if (nearObstacle(g.x, g.y, Math.max(5, (ydim/20)|0))) {
        draggingObstacle = true;
        dragStartX = g.x;
        dragStartY = g.y;
        const c = obsCenter();
        obsCenterX = c.x;
        obsCenterY = c.y;
        canvas.style.cursor = 'grabbing';
        // Dismiss tooltip
        const tip = document.getElementById('drag-tooltip');
        if (tip) tip.style.display = 'none';
      }
    } else {
      drawing = true;
      paint(g.x, g.y);
    }
  });

  canvas.addEventListener('pointermove', e => {
    const g = gxy(e.clientX, e.clientY);
    e.preventDefault();

    if (drawMode === 'move') {
      if (draggingObstacle) {
        const dx = g.x - dragStartX;
        const dy = g.y - dragStartY;
        if (Math.abs(dx) >= 1 || Math.abs(dy) >= 1) {
          moveObstacle(Math.round(dx), Math.round(dy));
          dragStartX = g.x;
          dragStartY = g.y;
        }
      } else {
        // Show grab cursor when hovering near obstacle
        canvas.style.cursor = nearObstacle(g.x, g.y, Math.max(5, (ydim/20)|0)) ? 'grab' : 'crosshair';
      }
    } else if (drawing) {
      paint(g.x, g.y);
    }
  });

  canvas.addEventListener('pointerup', () => { drawing = false; draggingObstacle = false; canvas.style.cursor = drawMode === 'move' ? 'default' : 'crosshair'; });
  canvas.addEventListener('pointerleave', () => { drawing = false; draggingObstacle = false; });
  canvas.addEventListener('touchstart', e=>e.preventDefault(), {passive:false});

  /* ── UI ────────────────────────────────────────────────── */
  const reSlider=document.getElementById('reynolds-slider'), reVal=document.getElementById('reynolds-value');
  const spSlider=document.getElementById('speed-slider'), spVal=document.getElementById('speed-value');
  const rsSlider=document.getElementById('resolution-slider'), rsVal=document.getElementById('resolution-value'), rsH=document.getElementById('resolution-height');

  reSlider.addEventListener('input',()=>{ reynolds=+reSlider.value; reVal.textContent=reynolds; updateOmega(); updateEdu(); });
  spSlider.addEventListener('input',()=>{ stepsPerFrame=+spSlider.value; spVal.textContent=stepsPerFrame; });
  rsSlider.addEventListener('input',()=>{ rsVal.textContent=rsSlider.value; rsH.textContent=Math.floor(rsSlider.value/2); });
  rsSlider.addEventListener('change',()=>{ xdim=+rsSlider.value; ydim=Math.floor(xdim/2); fullReset(); });

  document.querySelectorAll('.preset-btn').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('.preset-btn').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); currentPreset=b.dataset.preset;
    placePreset(currentPreset); initFlow(); if(vizMode==='streamlines')initParticles(); updateEdu();
  }));
  document.querySelectorAll('.viz-btn').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('.viz-btn').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); vizMode=b.dataset.viz; if(vizMode==='streamlines')initParticles(); updateEdu();
  }));
  document.querySelectorAll('.draw-btn').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('.draw-btn').forEach(x=>x.classList.remove('active'));
    b.classList.add('active'); drawMode=b.dataset.draw;
    canvas.style.cursor = drawMode === 'move' ? 'default' : 'crosshair';
  }));
  document.getElementById('pause-btn').addEventListener('click',()=>{ paused=!paused; document.getElementById('pause-btn').textContent=paused?'Play':'Pause'; });
  document.getElementById('reset-btn').addEventListener('click',()=>{ placePreset(currentPreset); initFlow(); if(vizMode==='streamlines')initParticles(); });
  document.getElementById('screenshot-btn').addEventListener('click',()=>{ const a=document.createElement('a'); a.download=`wind-tunnel-Re${reynolds}-${vizMode}.png`; a.href=canvas.toDataURL('image/png'); a.click(); });
  document.querySelectorAll('.panel-header').forEach(h=>h.addEventListener('click',()=>{
    const id=h.dataset.toggle; if(!id)return;
    const body = document.getElementById(id);
    body.classList.toggle('collapsed');
    body.classList.remove('mobile-collapse');
    h.classList.toggle('collapsed');
    h.classList.remove('mobile-collapse');
  }));
  function handleMobile() { if(window.innerWidth<=900) document.querySelectorAll('.panel-body').forEach(b=>{ b.classList.add('mobile-collapse','collapsed'); b.previousElementSibling?.classList.add('mobile-collapse','collapsed'); }); }
  window.addEventListener('resize', handleMobile);

  /* ── Full Reset ────────────────────────────────────────── */
  function fullReset() {
    allocArrays();
    canvas.width = xdim; canvas.height = ydim;
    imgData = ctx.createImageData(xdim, ydim);
    imgBuf = imgData.data;
    updateOmega();
    placePreset(currentPreset); initFlow();
    if (vizMode === 'streamlines') initParticles();
    updateEdu();
  }

  /* ── Main loop ─────────────────────────────────────────── */
  function loop() {
    if (!paused) {
      for (let s = 0; s < stepsPerFrame; s++) simulate();
      render();
    }
    requestAnimationFrame(loop);
  }

  fullReset();
  handleMobile();

  // Warmup: run ~3000 steps so vortex street is already visible on load
  // Do it in chunks to avoid blocking the UI for too long
  let warmupRemaining = 20000;
  function warmupLoop() {
    if (warmupRemaining > 0) {
      const chunk = Math.min(500, warmupRemaining);
      for (let i = 0; i < chunk; i++) simulate();
      warmupRemaining -= chunk;
      render();
      requestAnimationFrame(warmupLoop);
    } else {
      loop();
    }
  }
  warmupLoop();

  // Auto-dismiss drag tooltip after 4 seconds
  setTimeout(() => {
    const tip = document.getElementById('drag-tooltip');
    if (tip) { tip.style.opacity = '0'; setTimeout(() => tip.style.display = 'none', 500); }
  }, 4000);

})();
