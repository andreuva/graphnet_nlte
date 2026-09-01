import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ===================================================================
# 1. FIGURE CONFIGURATION & SETUP (Tightened vertical height)
# ===================================================================
FIG_WIDTH, FIG_HEIGHT = 38, 14
X_LIMIT, Y_LIMIT = 40, 15
DPI_SETTING = 300

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_SETTING)
ax.set_xlim(0, X_LIMIT)
ax.set_ylim(0, Y_LIMIT)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')


# ===================================================================
# 2. COLOR PALETTE
# ===================================================================
COL_TARGET   = '#D8B4FE'   # Soft Purple  – target node xi
COL_NEIGHBOR = '#BAE6FD'   # Soft Blue    – neighbor xj
COL_EDGE     = '#FED7AA'   # Soft Orange  – edge features
COL_HIDDEN   = '#E2E8F0'   # Light Gray   – hidden neurons
COL_LATENT   = '#FBCFE8'   # Soft Pink    – latent / message
COL_OUTPUT   = '#A7F3D0'   # Soft Emerald – output yi
COL_ARROW    = '#475569'   # Slate arrow color


# ===================================================================
# 3. TEXT LABELS (NON-MATH STRINGS)
# ===================================================================
TEXT_LEGEND_TITLE      = 'Neuron & Node Color Legend'
TEXT_LEGEND_TARGET     = 'Target Node'
TEXT_LEGEND_NEIGHBOR   = 'Neighbor Node'
TEXT_LEGEND_EDGE       = 'Edge Feature'
TEXT_LEGEND_HIDDEN     = 'Hidden Layer Neurons'
TEXT_LEGEND_LATENT     = 'Latent / Message'
TEXT_LEGEND_OUTPUT     = 'Output Prediction'

TEXT_STAGE1_TITLE      = '1. Input Graph'
TEXT_STAGE2_TITLE      = '2. Encoder Stage'
TEXT_NODE_ENCODING     = 'Node\nEncoding'
TEXT_EDGE_ENCODING     = 'Edge\nEncoding'

TEXT_NODE_ENCODER_TITLE= 'Node Encoder'
TEXT_EDGE_ENCODER_TITLE= 'Edge Encoder'

TEXT_STAGE3_TITLE      = '3. Processor — MetaLayer Step k  (repeated M = 8 times)'
TEXT_EDGE_MLP_TITLE    = 'Edge MLP'
TEXT_NODE_MLP1_TITLE   = 'Node MLP 1'
TEXT_NODE_MLP2_TITLE   = 'Node MLP 2'
TEXT_UPDATED_EDGE      = 'Updated\nEdge'
TEXT_SCATTER_MEAN      = 'scatter_mean\n(Aggregation)'
TEXT_RESIDUAL_ADD      = 'Residual\nAdd'

TEXT_STAGE4_TITLE      = '4. Decoder & Output Stage'
TEXT_DECODER_MLP_TITLE = 'Decoder MLP'
TEXT_OUTPUT_TITLE      = 'Output\nPopulations'


# ===================================================================
# 4. MATHEMATICAL EQUATIONS (LATEX STRINGS)
# ===================================================================
EQ_TARGET_NODE_SYM     = r'$x_i$'
EQ_NEIGHBOR_NODE_SYM   = r'$x_j$'
EQ_EDGE_FEATURE_SYM    = r'$e_{ij}$'
EQ_LATENT_SYM          = r'$h$'
EQ_OUTPUT_SYM          = r'$y_i$'

EQ_INPUT_GRAPH         = r'Nodes $x_i \in \mathbb{R}^{11}$,  Edges $e_{ij} \in \mathbb{R}^1$'
EQ_NODE_ENCODER        = r'$x_i \;\rightarrow\; h_{v,i}^{(0)} \in \mathbb{R}^{128}$'
EQ_EDGE_ENCODER        = r'$e_{ij} \;\rightarrow\; h_{e,ij}^{(0)} \in \mathbb{R}^{128}$'

EQ_EDGE_MLP_UPDATE     = r'$\tilde{h}_{e,ij}^{(k)} = \mathrm{EdgeMLP}\!\left([h_{v,i},\, h_{v,j},\, h_{e,ij},\, u]\right)$'
EQ_SCATTER_MEAN_LABEL  = r'$\frac{\sum}{N}$'
EQ_SCATTER_MEAN_FORMULA= r'$\bar{m}_i = \mathrm{mean}(m_{ij})$'
EQ_NODE_MLP2_UPDATE    = r'$\Delta h_{v,i} = \mathrm{NodeMLP2}\!\left([h_{v,i},\, \bar{m}_i,\, u]\right)$'
EQ_RESIDUAL_ADD_LABEL  = r'$+$'
EQ_RESIDUAL_ADD_FORMULA= r'$h_{v,i}^{(k)} = h_{v,i}^{(k\!-\!1)} + \Delta h_{v,i}$'

EQ_DECODER_TRANSFORM   = r'$h_{v,i}^{(M)} \;\rightarrow\; y_i \in \mathbb{R}^{6}$'
EQ_OUTPUT_DIM          = r'$y_i \in \mathbb{R}^{6}$'


# ===================================================================
# 5. DRAWING HELPER FUNCTIONS
# ===================================================================
def draw_circle(ax, x, y, r=0.40, color=COL_NEIGHBOR, label='', fs=20, lw=2.0):
    """Draws a node circle with optional mathematical symbol or text label inside."""
    ax.add_patch(patches.Circle((x, y), r, fc=color, ec='black', lw=lw, zorder=4))
    if label:
        ax.text(x, y, label, fontsize=fs, weight='bold', ha='center', va='center', zorder=5)

def draw_mlp(ax, x0, yc, sizes, colors, title='', w=2.8, dy=0.82, node_r=0.24):
    """Draws a fully-connected multi-layer perceptron architecture diagram."""
    xs = np.linspace(x0, x0 + w, len(sizes))
    coords = []
    for li, n in enumerate(sizes):
        ys = [yc + (i - (n-1)/2.0) * dy for i in range(n)] if n > 1 else [yc]
        coords.append([(xs[li], y) for y in ys])
    
    # Draw connections between adjacent layers
    for l in range(len(sizes)-1):
        for (xa, ya) in coords[l]:
            for (xb, yb) in coords[l+1]:
                ax.plot([xa, xb], [ya, yb], color='#94A3B8', lw=1.2, zorder=2)
                
    # Draw layer nodes
    for li, pts in enumerate(coords):
        cs = colors[li]
        for ni, (px, py) in enumerate(pts):
            c = cs[ni % len(cs)] if isinstance(cs, list) else cs
            draw_circle(ax, px, py, r=node_r, color=c, lw=1.4)
            
    # Draw component title above MLP
    if title:
        top = yc + max(sizes) * dy / 2.0 + 0.50
        ax.text(x0 + w/2.0, top, title, fontsize=20, weight='bold', ha='center', va='bottom')

def draw_arrow(x1, y1, x2, y2, txt='', fs=18):
    """Draws a block directional arrow with optional text label."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=4.5, color=COL_ARROW, mutation_scale=24), zorder=3)
    if txt:
        ang = np.degrees(np.arctan2(y2-y1, x2-x1))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.40, txt, fontsize=fs, weight='bold', ha='center', va='bottom',
                rotation=ang, zorder=6, color='#0F172A')


# ===================================================================
# 6. LEGEND RENDERING (Top Left)
# ===================================================================
legend_box = patches.FancyBboxPatch((0.4, 11.4), 8.8, 2.4, boxstyle='round,pad=0.25',
                                    lw=1.6, ec='#CBD5E1', fc='#F8FAFC', zorder=2)
ax.add_patch(legend_box)
ax.text(4.8, 13.3, TEXT_LEGEND_TITLE, fontsize=18, weight='bold', ha='center')

legend_items = [
    (0.8, 12.6, COL_TARGET,   TEXT_LEGEND_TARGET,   EQ_TARGET_NODE_SYM),
    (4.9, 12.6, COL_NEIGHBOR, TEXT_LEGEND_NEIGHBOR, EQ_NEIGHBOR_NODE_SYM),
    (0.8, 12.0, COL_EDGE,     TEXT_LEGEND_EDGE,     EQ_EDGE_FEATURE_SYM),
    (4.9, 12.0, COL_HIDDEN,   TEXT_LEGEND_HIDDEN,   ""),
    (0.8, 11.4, COL_LATENT,   TEXT_LEGEND_LATENT,   EQ_LATENT_SYM),
    (4.9, 11.4, COL_OUTPUT,   TEXT_LEGEND_OUTPUT,   EQ_OUTPUT_SYM),
]

for lx, ly, lc, text_str, eq_str in legend_items:
    draw_circle(ax, lx, ly + 0.15, r=0.18, color=lc, lw=1.2)
    full_label = f"{text_str} ({eq_str})" if eq_str else text_str
    ax.text(lx + 0.38, ly + 0.15, full_label, fontsize=14.5, weight='bold', ha='left', va='center', color='#1E293B')


# ===================================================================
# 7. STAGE 1: INPUT GRAPH
# ===================================================================
cx, cy = 4.0, 5.5
draw_circle(ax, cx, cy, r=0.60, color=COL_TARGET, label=EQ_TARGET_NODE_SYM, fs=22)

angles = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/6
for i, a in enumerate(angles):
    nx = cx + 2.2 * np.cos(a)
    ny = cy + 2.2 * np.sin(a)
    ax.plot([cx, nx], [cy, ny], color='black', lw=2.0, zorder=2)
    draw_circle(ax, nx, ny, r=0.50, color=COL_NEIGHBOR, label=f'$x_{{j{i+1}}}$', fs=20)

ax.text(cx, cy + 4.9, TEXT_STAGE1_TITLE, fontsize=19, weight='bold', ha='center', va='top', color='black')
# ax.text(cx, cy - 4.2, EQ_INPUT_GRAPH, fontsize=18, weight='bold', ha='center', va='top', color='#1E293B')


# ===================================================================
# 8. STAGE 2: ENCODER STAGE
# ===================================================================
ax.text(10.8, 10.2, TEXT_STAGE2_TITLE, fontsize=21, weight='bold', ha='center', color='#0F172A')

# Node Encoder Track
draw_arrow(6.6, 6.5, 9.0, 7.5, '')
ax.text(7.2, 7.8, TEXT_NODE_ENCODING, fontsize=18, weight='bold', ha='center', va='center', color='#0F172A')
draw_mlp(ax, 9.4, 7.5, [1, 4, 4, 1],
         [COL_TARGET, COL_HIDDEN, COL_HIDDEN, COL_LATENT],
         title=TEXT_NODE_ENCODER_TITLE, w=2.8, dy=0.65)
# ax.text(10.8, 8.8, EQ_NODE_ENCODER, fontsize=20, weight='bold', ha='center')

# Edge Encoder Track
draw_arrow(6.6, 4.5, 9.0, 3.5, '')
ax.text(7.2, 3.3, TEXT_EDGE_ENCODING, fontsize=18, weight='bold', ha='center', va='center', color='#0F172A')
draw_mlp(ax, 9.4, 3.5, [1, 4, 4, 1],
         [COL_EDGE, COL_HIDDEN, COL_HIDDEN, COL_LATENT],
         title=TEXT_EDGE_ENCODER_TITLE, w=2.8, dy=0.65)
# ax.text(10.8, 2.8, EQ_EDGE_ENCODER, fontsize=20, weight='bold', ha='center')


# ===================================================================
# 9. STAGE 3: PROCESSOR STAGE (Snug, tightened box & scaled MLPs)
# ===================================================================
Y_BASELINE = 6.5

# Tightened processor container box (Height = 8.6, spanning Y = 2.2 to 10.8)
processor_box = patches.FancyBboxPatch((13.4, 2.2), 18.0, 8.6, boxstyle='round,pad=0.3',
                                       lw=2.0, ec='#CBD5E1', fc='#FAFAFA', zorder=1)
ax.add_patch(processor_box)
ax.text(22.4, 10.1, TEXT_STAGE3_TITLE, fontsize=21, weight='bold', ha='center', color='#0F172A')

# Edge MLP Update (Enlarged dy = 0.82)
draw_mlp(ax, 14.2, Y_BASELINE, [3, 5, 5, 1],
         [[COL_TARGET, COL_NEIGHBOR, COL_EDGE], COL_HIDDEN, COL_HIDDEN, COL_LATENT],
         title=TEXT_EDGE_MLP_TITLE, w=2.8, dy=0.82)
# ax.text(15.6, Y_BASELINE - 3.6, EQ_EDGE_MLP_UPDATE, fontsize=19, weight='bold', ha='center', color='#0F172A')

# Arrow Edge -> Node
draw_arrow(17.2, Y_BASELINE, 18.6, Y_BASELINE, '')
ax.text(17.9, Y_BASELINE + 0.60, TEXT_UPDATED_EDGE, fontsize=17, weight='bold', ha='center', va='bottom')

# Node MLP 1 (Message calculation)
draw_mlp(ax, 18.9, Y_BASELINE, [2, 4, 4, 1],
         [[COL_NEIGHBOR, COL_LATENT], COL_HIDDEN, COL_HIDDEN, COL_LATENT],
         title=TEXT_NODE_MLP1_TITLE, w=2.4, dy=0.82)

# Connecting line -> scatter_mean
ax.plot([21.5, 22.2], [Y_BASELINE, Y_BASELINE], color='black', lw=2.0)

# Scatter Mean (Aggregation Operator)
draw_circle(ax, 22.9, Y_BASELINE, r=0.62, color='white', label=EQ_SCATTER_MEAN_LABEL, fs=22, lw=2.2)
ax.text(22.9, Y_BASELINE + 1.8, TEXT_SCATTER_MEAN, fontsize=18, weight='bold', ha='center')
# ax.text(22.9, Y_BASELINE - 1.8, EQ_SCATTER_MEAN_FORMULA, fontsize=19, weight='bold', ha='center')

# Connecting line -> Node MLP 2
ax.plot([23.52, 24.2], [Y_BASELINE, Y_BASELINE], color='black', lw=2.0)

# Node MLP 2 (State update - Enlarged dy = 0.82)
draw_mlp(ax, 24.4, Y_BASELINE, [3, 5, 5, 1],
         [[COL_TARGET, COL_LATENT, COL_EDGE], COL_HIDDEN, COL_HIDDEN, COL_LATENT],
         title=TEXT_NODE_MLP2_TITLE, w=2.8, dy=0.82)
# ax.text(25.8, Y_BASELINE - 3.6, EQ_NODE_MLP2_UPDATE, fontsize=19, weight='bold', ha='center', color='#0F172A')

# Connecting line -> Residual Add
ax.plot([27.4, 28.4], [Y_BASELINE, Y_BASELINE], color='black', lw=2.0)

# Residual Addition (+)
draw_circle(ax, 29.0, Y_BASELINE, r=0.52, color='white', label=EQ_RESIDUAL_ADD_LABEL, fs=24, lw=2.2)
ax.text(29.0, Y_BASELINE + 1.6, TEXT_RESIDUAL_ADD, fontsize=18, weight='bold', ha='center')
# ax.text(29.0, Y_BASELINE - 1.8, EQ_RESIDUAL_ADD_FORMULA, fontsize=19, weight='bold', ha='center')


# ===================================================================
# 10. STAGE 4: DECODER & OUTPUT STAGE
# ===================================================================
ax.text(35.1, 10.1, TEXT_STAGE4_TITLE, fontsize=21, weight='bold', ha='center', color='#0F172A')

draw_arrow(29.5, Y_BASELINE, 32.8, Y_BASELINE, '')

# Decoder MLP
draw_mlp(ax, 33.0, Y_BASELINE, [1, 4, 4, 1],
         [COL_LATENT, COL_HIDDEN, COL_HIDDEN, COL_OUTPUT],
         title=TEXT_DECODER_MLP_TITLE, w=2.4, dy=0.82)
# ax.text(34.2, Y_BASELINE - 3.0, EQ_DECODER_TRANSFORM, fontsize=20, weight='bold', ha='center')

# Connecting line -> Output Node
ax.plot([35.6, 36.6], [Y_BASELINE, Y_BASELINE], color='black', lw=2.2)

# Final Output Circle
draw_circle(ax, 37.3, Y_BASELINE, r=0.60, color=COL_OUTPUT, label=EQ_OUTPUT_SYM, fs=22)

# Separate text label ("Output Populations")
ax.text(37.3, Y_BASELINE - 1.5, TEXT_OUTPUT_TITLE, fontsize=18, weight='bold', ha='center')
# ax.text(37.3, Y_BASELINE - 3.0, EQ_OUTPUT_DIM, fontsize=20, weight='bold', ha='center')


# ===================================================================
# 11. SAVE OUTPUT DIAGRAM
# ===================================================================
out_local    = 'gnn_architecture_centered.png'
out_artifact = '/home/andreuva/.gemini/antigravity-ide/brain/be0cc8ae-a8e0-4065-8cbe-ce3250bb03bf/gnn_architecture_centered.png'

plt.tight_layout()
plt.savefig(out_local, dpi=300, facecolor='white', edgecolor='none')
plt.savefig(out_artifact, dpi=300, facecolor='white', edgecolor='none')
plt.close()

print(f'Centered scientific diagram created at: {out_local}')
