"""Independent exact binomial oracle; no WebPPL or simulator imports."""
from fractions import Fraction
from math import comb

def exact(case):
    n = case["hidden_dice"]
    own = case["own_dice"].count(case["face"])
    prior = sum((Fraction(comb(n,k)*5**(n-k),6**n)
                 for k in range(n+1) if k+own >= case["quantity"]), Fraction(0))
    a,b = (Fraction(1),Fraction(3,10)) if case["condition"] == "adversarial" else (Fraction(1,2),Fraction(1,2))
    evidence = prior*a + (1-prior)*b
    posterior = prior*a/evidence
    return {"prior":float(prior), "posterior":float(posterior),
            "prior_exact":str(prior), "posterior_exact":str(posterior),
            "bid_probability":float(evidence)}
