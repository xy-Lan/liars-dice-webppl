import json

def build_prompt(c):
    policy = ("P(emit this bid | claim true)=1; P(emit this bid | claim false)=0.3."
              if c["condition"] == "adversarial" else
              "P(emit this bid | claim true)=0.5; P(emit this bid | claim false)=0.5. The bid is independent of all dice.")
    return f"""Write only executable WebPPL source (no prose). This is a simplified single-bid Liar's Dice inference task, not full gameplay.
All hidden dice are independent fair six-sided dice. Ones are NOT wild.
Observer's known dice: {json.dumps(c['own_dice'])}.
Opponent has {c['hidden_dice']} hidden dice.
A fixed candidate claim was specified BEFORE any dice were rolled: at least {c['quantity']} dice show face {c['face']} across BOTH hands.
The bidder knows both hands (stylized omniscient signal policy). {policy}
We observed that this candidate bid WAS emitted. Condition on this observation. Do not treat the claim itself as true evidence.
Infer the probability that this same claim is true after observing the bid.
Use Infer({{method:'enumerate', model:function(){{ ... }}}}) returning the Boolean claim; represent each hidden die explicitly (uniformDraw([1,2,3,4,5,6]) or equivalent).
Condition using observe(Bernoulli({{p: likelihood}}), true) or factor(Math.log(likelihood)). Use WebPPL functional style, var, map/repeat/filter/reduce; no mutable assignments or loops.
After inference, print exactly one result line using:
console.log('RESULT_JSON:' + JSON.stringify({{probability: Math.exp(dist.score(true))}}));
where dist is your inferred Boolean distribution. No imports, file operations, or external packages. Do not hardcode a computed answer.
"""

def reference_program(c):
    # Enumerates concrete dice, independently of the oracle's binomial formula.
    own = c['own_dice'].count(c['face'])
    likelihood = '(truth ? 1 : 0.3)' if c['condition']=='adversarial' else '0.5'
    return f"""var dist = Infer({{method: 'enumerate', model: function() {{
  var dice = repeat({c['hidden_dice']}, function() {{ return uniformDraw([1,2,3,4,5,6]); }});
  var matches = filter(function(d) {{ return d === {c['face']}; }}, dice).length;
  var truth = matches + {own} >= {c['quantity']};
  observe(Bernoulli({{p: {likelihood}}}), true);
  return truth;
}}}});
console.log('RESULT_JSON:' + JSON.stringify({{probability: Math.exp(dist.score(true))}}));
"""
