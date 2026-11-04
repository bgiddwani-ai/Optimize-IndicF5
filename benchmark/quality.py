"""Paired audio and Hindi ASR evidence, outside timed performance runs."""
import hashlib,io,json,unicodedata
from pathlib import Path
import numpy as np
import requests
import soundfile as sf
import torch
from transformers import pipeline
import jiwer
from app.workload import TARGETS
from app.runtime import BACKENDS
from app.config import settings

root=Path('evidence');quality=root/'quality';quality.mkdir(parents=True,exist_ok=True)
integrity=[]
for backend in BACKENDS:
    directory=quality/backend;directory.mkdir(exist_ok=True)
    for i,text in enumerate(TARGETS):
        response=requests.post('http://127.0.0.1:8000/v1/audio/speech',json={'text':text,'backend':backend},timeout=600)
        response.raise_for_status();audio,sr=sf.read(io.BytesIO(response.content))
        assert sr==24000 and audio.size and np.isfinite(audio).all() and np.sqrt(np.mean(audio**2))>=0.001
        (directory/f'{i:02d}.wav').write_bytes(response.content)
        row={'backend':backend,'target':i,'samples':len(audio),'rms':float(np.sqrt(np.mean(audio**2))),
             'sha256':hashlib.sha256(response.content).hexdigest()}
        integrity.append(row);print(json.dumps(row),flush=True)
(quality/'integrity.json').write_text(json.dumps({'passed':True,'rows':integrity},indent=2))

def normalize(text):
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in unicodedata.normalize('NFC',text)).split())

revision='41f01f3fe87f28c78e2fbf8b568835947dd65ed9'
asr=pipeline('automatic-speech-recognition',model='openai/whisper-large-v3-turbo',revision=revision,torch_dtype=torch.float16,device='cuda:0')
results={}
for backend in BACKENDS:
    rows=[]
    for i,text in enumerate(TARGETS):
        hypothesis=asr(str(quality/backend/f'{i:02d}.wav'),generate_kwargs={'language':'hindi','task':'transcribe'})['text']
        rows.append({'index':i,'target':text,'hypothesis':hypothesis})
        print(json.dumps({'backend':backend,**rows[-1]},ensure_ascii=False),flush=True)
    refs=[normalize(r['target']) for r in rows];hyps=[normalize(r['hypothesis']) for r in rows]
    w=jiwer.process_words(refs,hyps);c=jiwer.process_characters(refs,hyps)
    results[backend]={'wer':w.wer,'cer':c.cer,'word_errors':w.substitutions+w.deletions+w.insertions,
                      'character_errors':c.substitutions+c.deletions+c.insertions,'rows':rows}
deltas={}
for candidate,baseline in [('bf16_eager','fp32_eager'),('bf16_aoti','fp32_eager'),('bf16_aoti','bf16_eager')]:
    dw=(results[candidate]['wer']-results[baseline]['wer'])*100
    dc=(results[candidate]['cer']-results[baseline]['cer'])*100
    deltas[candidate+'_minus_'+baseline]={'wer_delta_pp':dw,'cer_delta_pp':dc,
        'no_regression':dw<=1e-12 and dc<=1e-12}
passed=max(results[b]['cer'] for b in results)<=settings()["cer_ceiling"] and all(d['no_regression'] for d in deltas.values())
report={'passed':passed,'asr_model':'openai/whisper-large-v3-turbo','asr_revision':revision,'forced_language':'hindi',
        'normalization':'NFC, Unicode punctuation to space, whitespace collapse; CER includes spaces',
        'absolute_cer_ceiling':settings()['cer_ceiling'],'deltas':deltas,'human_listening':'not independently performed','results':results}
(quality/'asr.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'quality_gate_passed':passed,'deltas':deltas}),flush=True)
# Preserve performance evidence even if a finite candidate regresses in ASR.
# The report marks it unqualified; there is no automatic candidate promotion.
