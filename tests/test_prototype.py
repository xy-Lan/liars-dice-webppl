import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ldice.bedrock import BedrockClient, MODEL_ALIASES
from ldice.executor import execute, extract_source, parse_probability, resolve_webppl
from ldice.game import simulate, validate
from ldice.oracle import exact
from ldice.prompts import build_prompt, reference_program
from ldice.scoring import score
ROOT=Path(__file__).resolve().parents[1]
CASES=[json.loads(l) for l in (ROOT/'data/cases.jsonl').read_text().splitlines()]

class OfflineTests(unittest.TestCase):
    def test_eight_matched_pairs(self):
        self.assertEqual(len(CASES),16)
        self.assertEqual(len({c['id'] for c in CASES}),16)
        for a,b in zip(CASES[::2],CASES[1::2]):
            validate(a);validate(b)
            self.assertEqual({k:v for k,v in a.items() if k not in ('id','condition')},
                             {k:v for k,v in b.items() if k not in ('id','condition')})
            self.assertEqual((a['condition'],b['condition']),('nonadversarial','adversarial'))

    def test_base_regression(self):
        base=exact(CASES[0]); adv=exact(CASES[1])
        self.assertEqual(base['prior_exact'],'2/27')
        self.assertAlmostEqual(base['posterior'],2/27)
        self.assertEqual(adv['posterior_exact'],'4/19')
        self.assertAlmostEqual(adv['posterior'],.21052631578947367)

    def test_oracle_against_bruteforce_dice(self):
        # Independent concrete dice enumeration, not binomial coefficients.
        for c in CASES:
            weights=[];success=[]
            for dice in itertools.product(range(1,7),repeat=c['hidden_dice']):
                truth=(c['own_dice']+list(dice)).count(c['face'])>=c['quantity']
                w=(1 if truth else .3) if c['condition']=='adversarial' else .5
                weights.append(w);success.append(w if truth else 0)
            self.assertAlmostEqual(exact(c)['posterior'],sum(success)/sum(weights),places=11)

    def test_boundaries_and_independence(self):
        for c in CASES:
            e=exact(c)
            if c['condition']=='nonadversarial': self.assertEqual(e['prior'],e['posterior'])
            else: self.assertGreaterEqual(e['posterior'],e['prior'])
        self.assertEqual(exact(CASES[-1])['posterior'],0)
        self.assertEqual(exact(CASES[-3])['posterior'],1)
        zero=dict(CASES[0],hidden_dice=0)
        self.assertEqual(exact(zero)['posterior'],0)

    def test_simulator_policy(self):
        class Controlled:
            def __init__(self, roll, coin): self.roll=roll;self.coin=coin
            def randint(self,a,b): return self.roll
            def random(self): return self.coin
        adv=CASES[1]
        self.assertTrue(simulate(adv,Controlled(6,.999))['bid_emitted'])
        self.assertTrue(simulate(adv,Controlled(1,.299))['bid_emitted'])
        self.assertFalse(simulate(adv,Controlled(1,.3))['bid_emitted'])
        for roll in (1,6):
            self.assertTrue(simulate(CASES[0],Controlled(roll,.499))['bid_emitted'])
            self.assertFalse(simulate(CASES[0],Controlled(roll,.5))['bid_emitted'])
        a=simulate(adv,random.Random(44));b=simulate(adv,random.Random(44))
        self.assertEqual(a,b)

    def test_prompt_policy_no_answers(self):
        a=build_prompt(CASES[1]);n=build_prompt(CASES[0])
        self.assertIn('claim true)=1',a);self.assertIn('claim false)=0.3',a)
        self.assertIn('independent of all dice',n)
        for text in (a,n):
            self.assertIn('Ones are NOT wild',text)
            self.assertIn('both hands',text)
            self.assertNotIn('0.210526',text)
            self.assertNotIn('2/27',text)

    def test_scoring_and_reversal(self):
        self.assertTrue(score(.2,.2)['numeric_pass'])
        self.assertEqual(score(1,0)['regret'],2)
        self.assertEqual(score(.5,.5)['action'],'accept')
        self.assertEqual(score(.3,.5)['regret'],0)
        self.assertEqual(score(exact(CASES[2])['posterior'],exact(CASES[2])['posterior'])['action'],'challenge')
        self.assertEqual(score(exact(CASES[3])['posterior'],exact(CASES[3])['posterior'])['action'],'accept')
        for p in (float('nan'),float('inf'),-1,1.1,True):
            with self.assertRaises(ValueError): score(p,.5)

    def test_result_parser_and_extractor(self):
        self.assertEqual(parse_probability('noise\nRESULT_JSON:{"probability":0.25}\n'),.25)
        for text in ('', 'RESULT_JSON:{"probability":null}', 'RESULT_JSON:{"probability":true}',
                     'RESULT_JSON:{"probability":NaN}', 'RESULT_JSON:{"probability":2}',
                     'RESULT_JSON:{"probability":0}\nRESULT_JSON:{"probability":1}'):
            with self.assertRaises((ValueError,KeyError,TypeError)): parse_probability(text)
        self.assertEqual(extract_source('```javascript\nvar x = 1;\n```'),'var x = 1;')
        with self.assertRaises(ValueError): extract_source('```javascript\nvar x = 1;')

    def test_converse_contract_without_network(self):
        class Fake:
            def converse(self,**kw):
                self.kw=kw
                return {'output':{'message':{'content':[{'reasoningContent':{}},{'text':'var x=1;'}]}},
                        'usage':{'inputTokens':10,'outputTokens':5},'stopReason':'end_turn'}
        for alias,model_id in MODEL_ALIASES.items():
            fake=Fake();client=BedrockClient(alias,region='us-east-2',client=fake)
            result=client.generate('hello')
            self.assertEqual(fake.kw['modelId'],model_id)
            self.assertEqual(fake.kw['messages'],[{'role':'user','content':[{'text':'hello'}]}])
            self.assertEqual(fake.kw['inferenceConfig'],{'temperature':.5,'maxTokens':6400})
            self.assertEqual(result['text'],'var x=1;')
            self.assertEqual(result['usage']['inputTokens'],10)

    def test_converse_truncation_and_empty(self):
        for content,reason in (([], 'end_turn'),([{'text':'partial'}],'max_tokens')):
            class Fake:
                def converse(self,**kw):
                    return {'output':{'message':{'content':content}},'stopReason':reason}
            with self.assertRaises(RuntimeError): BedrockClient('claude-haiku',client=Fake()).generate('x')

    def test_executor_errors_and_timeout(self):
        with patch('ldice.executor.resolve_webppl',return_value='/fake/webppl'):
            with patch('ldice.executor.subprocess.run',side_effect=subprocess.TimeoutExpired('webppl',1)):
                with self.assertRaises(subprocess.TimeoutExpired): execute('bad',timeout=1)
            with patch('ldice.executor.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','syntax error')):
                with self.assertRaisesRegex(RuntimeError,'syntax error'): execute('bad')

    def test_dry_run_cli_without_api_or_webppl(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable,str(ROOT/'run.py'),'--out',tmp],check=True,capture_output=True)
            rows=[json.loads(l) for l in (Path(tmp)/'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),16)
            self.assertTrue(all(r['status']=='dry_run_only' and 'scores' not in r for r in rows))
            self.assertTrue(all(r['model_id'] is None for r in rows))

class WebPPLTests(unittest.TestCase):
    def test_all_reference_programs_against_oracle(self):
        try: resolve_webppl()
        except FileNotFoundError: self.skipTest('npm ci required for WebPPL integration check')
        for c in CASES:
            with self.subTest(case=c['id']):
                p=execute(reference_program(c))['probability']
                self.assertAlmostEqual(p,exact(c)['posterior'],places=10)

if __name__=='__main__': unittest.main()
