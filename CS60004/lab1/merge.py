from peft import PeftModel                                                          
from transformers import AutoModelForCausalLM, AutoTokenizer                        
import torch, os                                                                    
base = 'models/Qwen2.5-1.5B-Instruct'                                               
adapter = 'outputs/task3_rft/v0-0331142902/best'                                   
out = 'outputs/task3_rft_merged'                                                   
tokenizer = AutoTokenizer.from_pretrained(adapter)                                  
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,      
device_map='cpu')                                                                   
model = PeftModel.from_pretrained(model, adapter)                                   
model = model.merge_and_unload()                                                    
os.makedirs(out, exist_ok=True)                                                     
model.save_pretrained(out)                                                          
tokenizer.save_pretrained(out)                                                      
print('done')                                                                       