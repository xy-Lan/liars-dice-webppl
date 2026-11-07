"""Generative simulator. Roll once, then emit or withhold ONE fixed candidate bid."""
import random

def validate(c):
    if c["condition"] not in ("nonadversarial", "adversarial"):
        raise ValueError("unknown condition")
    if not isinstance(c["hidden_dice"],int) or not 0 <= c["hidden_dice"] <= 8:
        raise ValueError("hidden_dice must be 0..8")
    if not c["own_dice"] or any(type(d) is not int or not 1 <= d <= 6 for d in c["own_dice"]):
        raise ValueError("invalid own dice")
    if type(c["face"]) is not int or not 1 <= c["face"] <= 6 or type(c["quantity"]) is not int or c["quantity"] < 1:
        raise ValueError("invalid bid")
    if c.get("bid_observed") is not True:
        raise ValueError("this pilot conditions on bid_observed=true only")

def simulate(c, rng=None):
    validate(c)
    rng = rng or random.Random()
    hidden = [rng.randint(1,6) for _ in range(c["hidden_dice"])]
    truth = (c["own_dice"]+hidden).count(c["face"]) >= c["quantity"]
    likelihood = (1.0 if truth else 0.3) if c["condition"] == "adversarial" else 0.5
    return {"hidden_dice":hidden,"claim_true":truth,"bid_emitted":rng.random() < likelihood}
