import math

def score(predicted,target,tolerance=1e-6):
    if type(predicted) not in (int,float) or not math.isfinite(predicted) or not 0 <= predicted <= 1:
        raise ValueError('invalid prediction')
    error=abs(predicted-target)
    # Accept earns +1 for true and -1 for false. Challenge has opposite payoff.
    action='accept' if predicted >= 0.5 else 'challenge'
    optimal='accept' if target >= 0.5 else 'challenge'
    reward=(2*target-1)*(1 if action=='accept' else -1)
    return {'absolute_error':error,'bernoulli_tvd':error,'numeric_pass':error<=tolerance,
            'action':action,'optimal_action':optimal,
            'decision_correct':action==optimal,'regret':max(0.0,abs(2*target-1)-reward)}
