"""Render a source-audited technique matrix; no model or benchmark calls."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
ROWS = [('Cube Matmul for both QK and PV', ['—', '✓']), ('Cube / Vector stages on separate streams', ['—', '✓']), ('Event-driven cross-chunk pipeline', ['—', '✓']), ('Gather KV once; reuse across QK and PV', ['—', '✓']), ('Explicit L1 depth / L0 double-buffer tiling', ['—', '✓']), ('TQue-managed producer / consumer buffers', ['—', '✓']), ('ReduceSum API instead of a manual sum tree', ['✓*', '✓']), ('Double buffering and input prefetch', ['—', '✓']), ('Vector arithmetic / UB reuse / mixed precision', ['✓*', '✓'])]

def main():
    fig,ax=plt.subplots(figsize=(12,10))
    ax.set_xlim(0,12);ax.set_ylim(-1,len(ROWS)+5.5);ax.axis('off')
    xs=[8.3,10.4]
    labels=['sparse_attention A', 'sparse_attention B']
    top=len(ROWS)+5.2
    ax.plot([.1,11.9],[top,top],color='black',lw=1.8)
    for x,label in zip(xs,labels):ax.text(x,len(ROWS)+.65,label,rotation=90,ha='center',va='bottom',fontsize=11)
    ax.plot([.1,11.9],[len(ROWS)+.4]*2,color='black',lw=1)
    for i,(label,vals) in enumerate(ROWS):
        y=len(ROWS)-i-.15
        if vals is None:
            ax.add_patch(Rectangle((.1,y-.4),11.8,.8,color='#ededed',lw=0))
            ax.text(.3,y,label,va='center',fontsize=13,fontweight='bold')
        else:
            ax.text(.45,y,label,va='center',fontsize=11)
            for x,v in zip(xs,vals):ax.text(x,y,v,ha='center',va='center',fontsize=15)
    ax.plot([.1,11.9],[.25,.25],color='black',lw=1.2)
    fig.text(.04,.027,'✓ present in selected source    ✓* inherited from supplied seed    — not observed / not applicable',fontsize=10)
    fig.text(.04,.009,'Source-level evidence only; marks do not establish an isolated performance benefit.',fontsize=9)
    fig.subplots_adjust(left=.025,right=.975,bottom=.065,top=.98)
    for ext in ('png','pdf','svg'):fig.savefig(ROOT/f'optimization-techniques.{ext}',dpi=200)
    plt.close(fig)

if __name__=='__main__':main()
