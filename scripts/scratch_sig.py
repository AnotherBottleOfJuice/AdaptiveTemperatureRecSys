import sqlite3, ast, math
from statistics import mean

conn = sqlite3.connect("mlruns/mlflow.db"); c = conn.cursor()
METRICS = ["valid/recall","valid/ndcg","valid/hitrate","valid/coverage"]

def dset(params):
    """Dataset id from vocab size (present on all runs): 157162=Yandex, 156922=Amazon.
    157225 is an earlier/broken Yandex preprocessing -> excluded."""
    v = params.get("data_vocab_size")
    if v == "157162": return "yandex"
    if v == "156922": return "amazon"
    return None

def runs_for(exp_names, dataset_sub):
    out = []
    cur = conn.cursor()
    for name in exp_names:
        r = cur.execute("SELECT experiment_id FROM experiments WHERE name=?", (name,)).fetchone()
        if not r: continue
        eid = r[0]
        run_ids = [x[0] for x in cur.execute(
            "SELECT run_uuid FROM runs WHERE experiment_id=? AND lifecycle_stage='active'", (eid,)).fetchall()]
        for ru in run_ids:
            params = dict(cur.execute("SELECT key,value FROM params WHERE run_uuid=?", (ru,)).fetchall())
            if dataset_sub and dset(params) != dataset_sub: continue
            curves = {}
            for m in METRICS:
                vals = [v for (v,) in cur.execute("SELECT value FROM metrics WHERE run_uuid=? AND key=? ORDER BY step",(ru,m)).fetchall()]
                curves[m] = vals
            out.append((params, curves))
    return out

def label(p):
    cls = p.get("tau_class_name","")
    try: t = ast.literal_eval(p.get("tau_json_args","{}"))
    except: t = {}
    if cls == "ConstantTau": return f"Const tau={t.get('initial_tau')}"
    if cls == "MACLossTau": return f"MACLoss bt={t.get('base_threshold')},lc={t.get('linear_coeff')}"
    name = {"LinearTau":"Linear","CosTau":"Cos","CosPerUserTau":"CosPerUser",
            "ShiftedCosPerUserTau":"ShiftedCosPerUser","ParameterTau":"Param"}.get(cls, cls.replace("Tau",""))
    base = f"{name} min={t.get('tau_min')},max={t.get('tau_max')}"
    if t.get("shift") is not None: base += f",shift={t.get('shift')}"
    if t.get("scale") is not None: base += f",scale={t.get('scale')}"
    return base

def peak(cv,m):
    v = cv[m]; return max(v) if v else float("nan")

def collect(exp_names, dsub, want_label, lr_want=None):
    d = {}; dup = {}
    for p,cv in runs_for(exp_names,dsub):
        if label(p)!=want_label: continue
        if lr_want and p.get("learning_rate")!=lr_want: continue
        seed = p.get("training_seed") or p.get("training_dataset_seed")
        vals = {m:peak(cv,m) for m in METRICS}
        if seed in d:
            dup[seed] = dup.get(seed,1)+1
            if vals["valid/recall"] > d[seed]["valid/recall"]:
                d[seed] = vals  # keep best run for that seed
        else:
            d[seed] = vals
    if dup: print(f"   [note] duplicate runs per seed for '{want_label}': {dup} (kept best)")
    return d

def pstd(x):
    n=len(x); m=mean(x); return (sum((v-m)**2 for v in x)/n)**0.5
def sstd(x):
    n=len(x); m=mean(x); return (sum((v-m)**2 for v in x)/(n-1))**0.5 if n>1 else 0.0

def report(name, const_exps, dsub, base_label, base_lr, cfgs):
    print("="*78); print(name); print("="*78)
    base = collect(const_exps, dsub, base_label, base_lr)
    if not base:
        labs = sorted(set(label(p) for p,_ in runs_for(const_exps,dsub)))
        print("BASELINE NOT FOUND. available const labels:", labs); return
    bseeds = sorted(base)
    brec = [base[s]['valid/recall'] for s in bseeds]
    print(f"Baseline {base_label} lr={base_lr}: seeds={bseeds}")
    print(f"   recall/seed={[round(x,4) for x in brec]}  mean={mean(brec):.4f} std(pop)={pstd(brec):.4f}")
    for nm, exps, lab, lr in cfgs:
        d = collect(exps, dsub, lab, lr)
        if not d:
            labs = sorted(set(label(p) for p,_ in runs_for(exps,dsub)))
            print(f"\n{nm}: NOT FOUND ({lab}). available:", labs[:14]); continue
        seeds = sorted(set(base)&set(d))
        if len(seeds)<2:
            print(f"\n{nm}: only seeds {sorted(d)} overlap={seeds}"); continue
        srec=[d[s]['valid/recall'] for s in seeds]
        db=[d[s]['valid/recall']-base[s]['valid/recall'] for s in seeds]
        n=len(db); md=mean(db); se=sstd(db)/math.sqrt(n) if n>1 else 0
        t=md/se if se>0 else float('inf')
        pct=md/mean([base[s]['valid/recall'] for s in seeds])*100
        wins=sum(1 for x in db if x>0)
        print(f"\n{nm}  [{lab}] lr={lr}  n={n} seeds={seeds}")
        print(f"   recall/seed={[round(x,4) for x in srec]}")
        print(f"   Δrecall/seed={[round(x,5) for x in db]}  wins={wins}/{n}")
        print(f"   meanΔ={md:+.5f} ({pct:+.2f}%)  paired t({n-1})={t:.2f}"
              + ("  (t-crit_0.05 two-sided df2=4.30, df4=2.78)" if n in (3,5) else ""))

# ---------------- YANDEX ----------------
const_exps=["constant_tau_30e","constant_tau_30e_part2","constant_tau_30e_part3","constant_tau_30e_part4","constant_tau_30e_part5"]
cfgsY=[
 ("Linear",["linear_tau_30e","linear_tau_30e_part2"],"Linear min=0.035,max=0.06","0.003"),
 ("Cos",["cos_tau_30e","cos_tau_30e_part2"],"Cos min=0.03,max=0.06","0.003"),
 ("Param",["param_tau_30e"],"Param min=0.035,max=0.06","0.003"),
 ("CosPerUser",["cos_per_user_tau_30e"],"CosPerUser min=0.045,max=0.065","0.003"),
 ("ShiftedCosPerUser",["shifted_cos_per_user_tau_30e","shifted_cos_per_user_tau_30e_part2","shifted_cos_per_user_tau_30e_part3"],"ShiftedCosPerUser min=0.04,max=0.06,shift=-0.45,scale=0.3","0.003"),
 ("MACLoss",["macloss_tau_30e"],"MACLoss bt=0.23,lc=-0.2","0.003"),
]
report("YANDEX (Yambda)", const_exps, "yandex", "Const tau=0.045", "0.003", cfgsY)

# ---------------- AMAZON ----------------
cfgsA=[
 ("Linear",["linear_tau_30e","linear_tau_30e_part2"],"Linear min=0.06,max=0.07","0.002"),
 ("Cos",["cos_tau_30e","cos_tau_30e_part2"],"Cos min=0.06,max=0.065","0.002"),
 ("Param",["param_tau_30e"],"Param min=0.06,max=0.075","0.002"),
 ("CosPerUser",["cos_per_user_tau_30e"],"CosPerUser min=0.06,max=0.065","0.002"),
 ("MACLoss",["macloss_tau_30e"],"MACLoss bt=0.32,lc=-0.05","0.002"),
]
report("AMAZON Beauty", const_exps, "amazon", "Const tau=0.065", "0.002", cfgsA)

# ================= CLEAN SUMMARY with p-values + Coverage =================
from scipy import stats
def summary(name, const_exps, dsub, base_label, base_lr, cfgs):
    print("\n"+"#"*78); print("# SUMMARY "+name); print("#"*78)
    base=collect(const_exps,dsub,base_label,base_lr)
    print(f"{'strategy':20} {'recall(mean)':12} {'Δ%':>7} {'wins':>5} {'paired_t':>9} {'p1side':>7} {'p2side':>7} {'cover':>7}")
    bcov=mean([base[s]['valid/coverage'] for s in base]); brec=mean([base[s]['valid/recall'] for s in base])
    print(f"{'BASELINE':20} {brec:.4f}       {'+0.0':>7} {'-':>5} {'-':>9} {'-':>7} {'-':>7} {bcov:.4f}")
    for nm,exps,lab,lr in cfgs:
        d=collect(exps,dsub,lab,lr)
        seeds=sorted(set(base)&set(d))
        if len(seeds)<2: print(f"{nm:20} (n/a)"); continue
        b=[base[s]['valid/recall'] for s in seeds]; x=[d[s]['valid/recall'] for s in seeds]
        t,p2=stats.ttest_rel(x,b)
        md=mean(x)-mean(b); pct=md/mean(b)*100
        wins=sum(1 for s in seeds if d[s]['valid/recall']>base[s]['valid/recall'])
        p1=p2/2 if t>0 else 1-p2/2
        cov=mean([d[s]['valid/coverage'] for s in seeds])
        print(f"{nm:20} {mean(x):.4f}       {pct:>+7.2f} {f'{wins}/{len(seeds)}':>5} {t:>9.2f} {p1:>7.3f} {p2:>7.3f} {cov:.4f}")

summary("YANDEX", const_exps, "yandex", "Const tau=0.045", "0.003", cfgsY)
summary("AMAZON", const_exps, "amazon", "Const tau=0.065", "0.002", cfgsA)
