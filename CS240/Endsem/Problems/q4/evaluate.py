"""
Public evaluator for Q4.  Run: python3 evaluate.py
"""
import time, numpy as np
from student_loss import (load_data, estimate_y_true, rbf_kernel,
                          fit_krr, predict_krr, model_loss,
                          predict_loss, fit_linear_predictor)
TIMEOUT = 120.0

def sep(t): print(f"\n{'='*55}\n  {t}\n{'='*55}")

def main():
    ts = time.time()

    sep("Part A: load_data")
    data = load_data("dataset.npz")
    req = {"x_train","y_train","x_test","y_test","x_cal","y_pred_cal","loss_cal"}
    if req - set(data.keys()):
        print(f"  FAIL — missing: {req-set(data.keys())}"); return
    print(f"  train={data['x_train'].shape} cal={data['x_cal'].shape}  OK")

    sep("Part B: estimate_y_true")
    t0 = time.time()
    yt_est = estimate_y_true(data["x_cal"], data["y_pred_cal"],
                              data["loss_cal"], data["x_cal"])
    r_cal = np.abs(yt_est - data["y_pred_cal"])
    print(f"  time={time.time()-t0:.2f}s  r range=[{r_cal.min():.3f},{r_cal.max():.3f}]  OK")

    sep("Part C: KRR + loss model")
    X = np.array([[0,0],[1,0],[0,1]], dtype=float)
    K = rbf_kernel(X, X, 1.0)
    print(f"  kernel diag={[f'{v:.3f}' for v in np.diag(K)]}")
    w = fit_krr(np.eye(3), np.array([1.,2.,3.]), 0.01)
    p = predict_krr(np.eye(3), w)
    print(f"  KRR sanity: targets=[1,2,3] preds=[{','.join(f'{v:.2f}' for v in p)}]")

    t0 = time.time()
    md = model_loss(r_cal, data["x_cal"], data["loss_cal"])
    Lp = predict_loss(md, r_cal, data["x_cal"])
    mse = float(np.mean((data["loss_cal"] - Lp)**2))
    q = "EXCELLENT" if mse<0.005 else "GOOD" if mse<0.05 else "PARTIAL" if mse<0.5 else "POOR"
    print(f"  model time={time.time()-t0:.2f}s  MSE={mse:.6f} ({q})  OK")

    sep("Part D: fit_linear_predictor")
    t0 = time.time()
    res = fit_linear_predictor(data["x_train"], data["y_train"], md)
    print(f"  time={time.time()-t0:.2f}s")
    if res and res.get("a") is not None:
        a, b = res["a"], res["b"]
        c = np.polyfit(data["x_train"], data["y_train"], 1)
        r_y = np.abs(data["y_test"]-(a*data["x_test"]+b))
        r_o = np.abs(data["y_test"]-np.polyval(c, data["x_test"]))
        Ly = float(np.mean(predict_loss(md, r_y, data["x_test"])))
        Lo = float(np.mean(predict_loss(md, r_o, data["x_test"])))
        tag = "beats OLS!" if Ly<Lo*0.99 else "comparable" if Ly<Lo*1.02 else "worse than OLS"
        print(f"  a={a:.4f} b={b:.4f}  yours={Ly:.4f} OLS={Lo:.4f}  {tag}  OK")

    sep("Summary")
    print(f"  Total: {time.time()-ts:.2f}s  (limit: {TIMEOUT}s)")

if __name__ == "__main__":
    main()
