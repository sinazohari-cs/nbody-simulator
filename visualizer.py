import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import matplotlib.patches as mpatches
import numpy as np
from physics import rk4_step, total_energy, center_of_mass, total_momentum, G
from bodies import Body

# ── constants ───────────────────────────────────────────────────────────────
TRAIL_LENGTH   = 400
COLORS = ['#E24B4A','#378ADD','#1D9E75','#EF9F27','#7F77DD','#D4537E','#97C459','#D85A30','#85B7EB','#5DCAA5']
BG             = '#080810'
PANEL_BG       = '#0e0e18'
GRID_COLOR     = '#12121e'
ACCENT         = '#2a2a3a'
TEXT_DIM       = '#44445a'
TEXT_MID       = '#7777aa'
TEXT_BRIGHT    = '#ccccee'
DEFAULT_MASS   = 1e30
DEFAULT_DT     = 3600

# ── helpers ──────────────────────────────────────────────────────────────────
def mass_to_size(mass, all_masses):
    mn, mx = min(all_masses), max(all_masses)
    if mx == mn:
        return 90
    n = (np.log10(mass) - np.log10(mn)) / (np.log10(mx) - np.log10(mn) + 1e-12)
    return 25 + n * 220

def fmt_mass(m):
    if m < 1e3:   return f'{m:.2f}'
    exp = int(np.floor(np.log10(abs(m))))
    coef = m / 10**exp
    return f'{coef:.1f}×10^{exp}'

def fmt_energy(e):
    if e == 0: return '0'
    exp = int(np.floor(np.log10(abs(e))))
    coef = e / 10**exp
    return f'{coef:.2f}×10^{exp} J'

def styled_button(fig, rect, label, fg='#aaaacc', hover='#1a1a2e'):
    ax = fig.add_axes(rect)
    ax.set_facecolor(PANEL_BG)
    btn = widgets.Button(ax, label, color=PANEL_BG, hovercolor=hover)
    btn.label.set_color(fg)
    btn.label.set_fontsize(8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(ACCENT)
        spine.set_linewidth(0.5)
    return btn, ax

# ── main ─────────────────────────────────────────────────────────────────────
def run_simulation(initial_bodies=None, dt=DEFAULT_DT, steps=500000,
                   xlim=(-3e11,3e11), ylim=(-3e11,3e11), use_normalized=False):

    bodies        = list(initial_bodies) if initial_bodies else []
    history       = [[] for _ in bodies]
    speed_history = [[] for _ in bodies]   # speed over time per body
    paused        = [False]
    follow_com    = [False]
    show_vectors  = [False]
    show_grid     = [True]
    press_coords  = [None]
    drag_line     = [None]
    current_mass  = [DEFAULT_MASS if not use_normalized else 1.0]
    initial_energy= [None]
    energy_history= []
    xlim_ref      = [xlim]
    ylim_ref      = [ylim]
    frame_n       = [0]
    elapsed_time  = [0.0]   # simulated seconds
    use_norm_ref  = [use_normalized]
    dt_ref        = [dt]

    # ── figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # simulation canvas
    ax_sim = fig.add_axes([0.07, 0.09, 0.62, 0.89])
    ax_sim.set_facecolor(BG)
    ax_sim.tick_params(colors=TEXT_DIM, labelsize=6)
    for sp in ax_sim.spines.values():
        sp.set_edgecolor(ACCENT)
        sp.set_linewidth(0.4)

    # right panel geometry
    PX, PW = 0.71, 0.275
    y = 0.97   # cursor, top-down

    def section(label, height=0.006):
        nonlocal y
        y -= height
        fig.text(PX, y, label.upper(), color=TEXT_DIM, fontsize=6.5,
                 fontweight='bold', va='top', fontfamily='monospace')
        y -= 0.008

    def add_axes(h, gap=0.008):
        nonlocal y
        y -= h
        ax = fig.add_axes([PX, y, PW, h])
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(ACCENT); sp.set_linewidth(0.4)
        y -= gap
        return ax

    # ── energy chart ─────────────────────────────────────────────────────────
    section('Energy Conservation')
    ax_energy = add_axes(0.18, gap=0.015)
    ax_energy.tick_params(colors=TEXT_DIM, labelsize=6)
    ax_energy.set_ylabel('drift %', color=TEXT_DIM, fontsize=6)
    ax_energy.set_xlabel('frames',  color=TEXT_DIM, fontsize=6)

    # ── stats readout ─────────────────────────────────────────────────────────
    section('Simulation Stats')
    ax_stats = add_axes(0.09, gap=0.015)
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.02, 0.92, '', color=TEXT_BRIGHT, fontsize=7.5,
                               va='top', fontfamily='monospace',
                               transform=ax_stats.transAxes)

    # ── mass slider ───────────────────────────────────────────────────────────
    section('New Body Mass')
    ax_mass = add_axes(0.032, gap=0.018)
    if use_norm_ref[0]:
        mass_slider = widgets.Slider(ax_mass, '', 0.1, 10.0,   valinit=1.0,   color='#378ADD')
    else:
        mass_slider = widgets.Slider(ax_mass, '', 1e27, 2e31, valinit=DEFAULT_MASS, color='#378ADD')
    mass_slider.valtext.set_color(TEXT_MID); mass_slider.valtext.set_fontsize(7)
    mass_slider.label.set_color(TEXT_DIM)
    ax_mass.set_facecolor(PANEL_BG)

    def on_mass_change(val):
        current_mass[0] = mass_slider.val
    mass_slider.on_changed(on_mass_change)

    # ── toggle buttons ────────────────────────────────────────────────────────
    section('Display Options')
    TBW = (PW - 0.006) / 3
    toggles = []
    toggle_labels = ['follow CoM', 'vel vectors', 'grid']
    toggle_states = [follow_com, show_vectors, show_grid]
    for ti, tlabel in enumerate(toggle_labels):
        bx = PX + ti*(TBW+0.003)
        tax = fig.add_axes([bx, y-0.032, TBW, 0.032])
        tax.set_facecolor(PANEL_BG)
        for sp in tax.spines.values(): sp.set_edgecolor(ACCENT); sp.set_linewidth(0.4)
        tb = widgets.Button(tax, tlabel, color=PANEL_BG, hovercolor='#18182a')
        tb.label.set_color('#1D9E75'); tb.label.set_fontsize(7.5)
        toggles.append(tb)
    y -= 0.042

    def make_toggle(idx, state_ref, btn):
        def _toggle(event):
            state_ref[0] = not state_ref[0]
            btn.label.set_color('#1D9E75' if state_ref[0] else TEXT_DIM)
        return _toggle
    for ti, (tb, sr) in enumerate(zip(toggles, toggle_states)):
        tb.on_clicked(make_toggle(ti, sr, tb))
    # initialise grid toggle colour
    toggles[2].label.set_color('#1D9E75')

    # ── presets ───────────────────────────────────────────────────────────────
    section('Presets', height=0.01)
    y -= 0.005
    PRESET_DEFS = [
        ('Figure-8 Orbit',     'figure_eight',  (-2,2),        (-2,2),        0.0005,   True),
        ('Binary Star',        'binary_star',   (-3e11,3e11),  (-3e11,3e11),  3600,     False),
        ('Solar System',       'solar_system',  (-8.5e11,8.5e11),(-8.5e11,8.5e11), 3600*12, False),
        ('Chaos 3-Body',       'chaos_three',   (-3e11,3e11),  (-3e11,3e11),  3600,     False),
        ('Lagrange / Trojan',  'lagrange_points',(-3e11,3e11), (-3e11,3e11),  3600*2,   False),
    ]
    preset_buttons = []
    for pi, (plabel, _, _, _, _, _) in enumerate(PRESET_DEFS):
        bax = fig.add_axes([PX, y-0.035, PW, 0.030])
        bax.set_facecolor(PANEL_BG)
        for sp in bax.spines.values(): sp.set_edgecolor(ACCENT); sp.set_linewidth(0.4)
        pb = widgets.Button(bax, plabel, color=PANEL_BG, hovercolor='#16162a')
        pb.label.set_color(TEXT_BRIGHT); pb.label.set_fontsize(8)
        preset_buttons.append(pb)
        y -= 0.040

    # ── control buttons ───────────────────────────────────────────────────────
    y -= 0.008
    HBW = (PW-0.004)/2
    btn_pause, _ = styled_button(fig, [PX,          y-0.038, HBW, 0.034], '⏸  Pause')
    btn_clear, _ = styled_button(fig, [PX+HBW+0.004,y-0.038, HBW, 0.034], '✕  Clear', fg='#E24B4A', hover='#1a0e0e')
    y -= 0.048

    # ── hint text ─────────────────────────────────────────────────────────────
    fig.text(0.01, 0.04, 'drag to set velocity  |  space = pause  |  scroll = zoom (use toolbar)',
             color=TEXT_DIM, fontsize=7.5)
    body_count_text = fig.text(0.36, 0.04, '', color=TEXT_MID, fontsize=8, ha='center')
    sim_time_text   = fig.text(0.01, 0.01, '', color=TEXT_DIM, fontsize=7, fontfamily='monospace')

    # ── title ─────────────────────────────────────────────────────────────────
    fig.text(PX, 0.995, 'N-Body Gravitational Simulator', color=TEXT_BRIGHT,
             fontsize=10, fontweight='bold', va='top')
    fig.text(PX, 0.975, 'RK4 · Newtonian gravity · real-time', color=TEXT_DIM,
             fontsize=7, va='top', fontstyle='italic')

    # ── preset loader ─────────────────────────────────────────────────────────
    from presets import figure_eight, binary_star, solar_system, chaos_three, lagrange_points
    PRESET_FNS = [figure_eight, binary_star, solar_system, chaos_three, lagrange_points]

    def load_preset(idx):
        _, fn_name, xl, yl, pdt, norm = PRESET_DEFS[idx]
        fn = PRESET_FNS[idx]
        new_bodies = fn()
        bodies.clear(); bodies.extend(new_bodies)
        history.clear(); history.extend([[] for _ in bodies])
        speed_history.clear(); speed_history.extend([[] for _ in bodies])
        energy_history.clear(); initial_energy[0] = None
        dt_ref[0] = pdt; use_norm_ref[0] = norm
        xlim_ref[0] = xl; ylim_ref[0] = yl
        elapsed_time[0] = 0.0
        if norm:
            mass_slider.valmin=0.1; mass_slider.valmax=10.0; mass_slider.set_val(1.0)
        else:
            mass_slider.valmin=1e27; mass_slider.valmax=2e31; mass_slider.set_val(DEFAULT_MASS)
        current_mass[0] = mass_slider.val
        frame_n[0] = 0  # triggers limit reset on next frame

    for pi in range(len(PRESET_DEFS)):
        preset_buttons[pi].on_clicked(lambda e, i=pi: load_preset(i))

    def on_clear(e):
        bodies.clear(); history.clear(); speed_history.clear()
        energy_history.clear(); initial_energy[0]=None; elapsed_time[0]=0.0
    btn_clear.on_clicked(on_clear)

    def on_pause(e):
        paused[0] = not paused[0]
        btn_pause.label.set_text('▶  Resume' if paused[0] else '⏸  Pause')
        btn_pause.label.set_color('#EF9F27' if paused[0] else TEXT_BRIGHT)
    btn_pause.on_clicked(on_pause)

    # ── mouse interaction ─────────────────────────────────────────────────────
    def on_press(event):
        if event.inaxes != ax_sim: return
        if event.xdata is not None:
            press_coords[0] = (event.xdata, event.ydata)

    def on_release(event):
        if event.inaxes != ax_sim: return
        if event.xdata is None or press_coords[0] is None: return
        px, py = press_coords[0]
        dx, dy = event.xdata - px, event.ydata - py
        span = xlim_ref[0][1] - xlim_ref[0][0]
        if abs(dx)+abs(dy) < span*0.008:
            press_coords[0] = None; return
        velocity = np.array([dx, dy]) / (span*0.5)
        if not use_norm_ref[0]:
            velocity *= 3e4
        color = COLORS[len(bodies) % len(COLORS)]
        nb = Body(mass=current_mass[0], position=np.array([px,py]),
                  velocity=velocity, color=color, name=f'Body {len(bodies)+1}')
        bodies.append(nb); history.append([]); speed_history.append([])
        press_coords[0] = None
        if drag_line[0]:
            try: drag_line[0].remove()
            except: pass
            drag_line[0] = None

    def on_motion(event):
        if event.inaxes != ax_sim or press_coords[0] is None: return
        if drag_line[0]:
            try: drag_line[0].remove()
            except: pass
        px, py = press_coords[0]
        if event.xdata is not None:
            line, = ax_sim.plot([px, event.xdata], [py, event.ydata],
                                color='#ffffff', alpha=0.4, lw=1, ls='--')
            # arrowhead
            drag_line[0] = line

    def on_key(event):
        if event.key == ' ': on_pause(None)

    fig.canvas.mpl_connect('button_press_event',   on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event',  on_motion)
    fig.canvas.mpl_connect('key_press_event',       on_key)

    # ── animation ─────────────────────────────────────────────────────────────
    def update(frame):
        if not paused[0] and bodies:
            rk4_step(bodies, dt_ref[0])
            elapsed_time[0] += dt_ref[0]
            for i, b in enumerate(bodies):
                if i < len(history):
                    history[i].append(b.position.copy())
                    if len(history[i]) > TRAIL_LENGTH: history[i].pop(0)
                if i < len(speed_history):
                    speed_history[i].append(np.linalg.norm(b.velocity))
                    if len(speed_history[i]) > 200: speed_history[i].pop(0)
            E = total_energy(bodies)
            energy_history.append(E)
            if initial_energy[0] is None and len(energy_history) > 5:
                initial_energy[0] = E
            if len(energy_history) > 600: energy_history.pop(0)

        # ── CoM offset ───────────────────────────────────────────────────────
        if follow_com[0] and bodies:
            com = center_of_mass(bodies)
            ox, oy = com
        else:
            ox, oy = 0.0, 0.0

        # ── draw simulation ───────────────────────────────────────────────────
        # preserve toolbar zoom/pan state across frames
        cur_xl = ax_sim.get_xlim()
        cur_yl = ax_sim.get_ylim()
        is_first_frame = frame_n[0] == 0

        ax_sim.clear()
        ax_sim.set_facecolor(BG)

        if is_first_frame:
            xl, yl = xlim_ref[0], ylim_ref[0]
            ax_sim.set_xlim(xl[0], xl[1])
            ax_sim.set_ylim(yl[0], yl[1])
        else:
            ax_sim.set_xlim(cur_xl)
            ax_sim.set_ylim(cur_yl)
        ax_sim.tick_params(colors=TEXT_BRIGHT, labelsize=8, length=4, width=0.5)
        for sp in ax_sim.spines.values(): sp.set_edgecolor(ACCENT); sp.set_linewidth(0.4)

        # unit label depends on mode
        unit = 'm' if use_norm_ref[0] else 'AU'
        scale = 1.0 if use_norm_ref[0] else 1/1.496e11

        def fmt_tick(val, _):
            v = val * scale
            if use_norm_ref[0]:
                return f'{v:.1f}'
            if abs(v) < 0.01:
                return '0'
            return f'{v:.2f}'

        import matplotlib.ticker as ticker
        ax_sim.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_tick))
        ax_sim.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_tick))
        ax_sim.set_xlabel(f'x ({unit})', color=TEXT_MID, fontsize=8)
        ax_sim.set_ylabel(f'y ({unit})', color=TEXT_MID, fontsize=8)

        if show_grid[0]:
            ax_sim.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle='-', alpha=1.0)
            ax_sim.set_axisbelow(True)

        all_masses = [b.mass for b in bodies] if bodies else [1.0]

        for i, b in enumerate(bodies):
            color = b.color or COLORS[i % len(COLORS)]
            pos = b.position - np.array([ox, oy]) if not follow_com[0] else b.position

            # fading trail
            if i < len(history) and len(history[i]) > 1:
                trail = np.array(history[i])
                if follow_com[0]:
                    com_hist = np.mean(trail, axis=0) if len(trail) else np.zeros(2)
                n = len(trail)
                for j in range(1, n):
                    alpha = (j/n)**1.5 * 0.85
                    tp = trail[j-1:j+1] - np.array([ox, oy]) if not follow_com[0] else trail[j-1:j+1]
                    ax_sim.plot(tp[:,0], tp[:,1], color=color, alpha=alpha, lw=0.9)

            # body dot
            sz = mass_to_size(b.mass, all_masses)
            ax_sim.scatter(*pos, s=sz, color=color, zorder=6, edgecolors='white',
                           linewidths=0.3, alpha=0.95)

            # label
            ax_sim.text(pos[0], pos[1], f'  {b.name}', color=color,
                        fontsize=7, alpha=0.75, va='center', fontfamily='monospace')

            # velocity vector
            if show_vectors[0]:
                span = xl[1]-xl[0]
                vscale = span * 0.08 / (np.linalg.norm(b.velocity)+1e-30)
                vend = pos + b.velocity * vscale
                ax_sim.annotate('', xy=vend, xytext=pos,
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.0, alpha=0.6))

        # CoM marker
        if follow_com[0] and bodies:
            ax_sim.scatter(0, 0, s=30, color='white', marker='+', zorder=7, alpha=0.4)

        # ── energy chart ─────────────────────────────────────────────────────
        ax_energy.clear()
        ax_energy.set_facecolor(PANEL_BG)
        ax_energy.tick_params(colors=TEXT_DIM, labelsize=6)
        ax_energy.set_ylabel('drift %', color=TEXT_DIM, fontsize=6)
        ax_energy.set_xlabel('frames',  color=TEXT_DIM, fontsize=6)
        for sp in ax_energy.spines.values(): sp.set_edgecolor(ACCENT); sp.set_linewidth(0.4)
        ax_energy.set_facecolor(PANEL_BG)

        if len(energy_history) > 3 and initial_energy[0] is not None:
            E0 = initial_energy[0]
            if abs(E0) > 0:
                drift = [(e-E0)/abs(E0)*100 for e in energy_history]
                ax_energy.plot(drift, color='#1D9E75', lw=0.9)
                ax_energy.axhline(0, color=ACCENT, lw=0.5, ls='--')
                md = max(abs(d) for d in drift)
                pad = max(md*0.15, 1e-4)
                ax_energy.set_ylim(-md-pad, md+pad)
                last = drift[-1]
                col = '#1D9E75' if abs(last)<0.1 else ('#EF9F27' if abs(last)<1.0 else '#E24B4A')
                ax_energy.text(0.97, 0.95, f'{last:+.4f}%', transform=ax_energy.transAxes,
                               color=col, fontsize=7, ha='right', va='top', fontfamily='monospace')

        # ── stats ─────────────────────────────────────────────────────────────
        if bodies:
            n = len(bodies)
            E = energy_history[-1] if energy_history else 0
            p = total_momentum(bodies)
            pmag = np.linalg.norm(p)
            com = center_of_mass(bodies)

            # elapsed time formatting
            secs = elapsed_time[0]
            if secs < 3600:
                tstr = f'{secs:.0f} s'
            elif secs < 86400:
                tstr = f'{secs/3600:.2f} hr'
            elif secs < 86400*365:
                tstr = f'{secs/86400:.2f} days'
            else:
                tstr = f'{secs/86400/365.25:.2f} yr'

            stats = (
                f'bodies    : {n}\n'
                f'sim time  : {tstr}\n'
                f'energy    : {fmt_energy(E)}\n'
                f'|momentum|: {pmag:.3e}\n'
                f'CoM x     : {com[0]:.3e}\n'
                f'CoM y     : {com[1]:.3e}'
            )
        else:
            stats = 'no bodies\n\nclick + drag\nto add one'

        ax_stats.clear(); ax_stats.axis('off')
        ax_stats.set_facecolor(PANEL_BG)
        ax_stats.text(0.04, 0.94, stats, color=TEXT_BRIGHT, fontsize=7.5,
                      va='top', fontfamily='monospace', transform=ax_stats.transAxes,
                      linespacing=1.7)

        body_count_text.set_text(
            f'{len(bodies)} {"body" if len(bodies)==1 else "bodies"}  ·  frame {frame_n[0]}')
        frame_n[0] += 1

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=16, blit=False)
    plt.show()