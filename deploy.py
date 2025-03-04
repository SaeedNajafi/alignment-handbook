from transformers import AutoTokenizer
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import LlamaForCausalLM

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("base_model_path", "/model-weights/Llama-3.2-1B-Instruct", "base model path")
flags.DEFINE_string("peft_model_path", "peft_model_path", "peft model path")

def main(argv):
    del argv
    
    model = LlamaForCausalLM.from_pretrained(FLAGS.base_model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.peft_model_path, padding_side="left")
    tokenizer.save_pretrained(FLAGS.peft_model_path+"_full_model")
    peft_model = PeftModel.from_pretrained(model, FLAGS.peft_model_path, is_trainable=False)
    peft_model = peft_model.bfloat16()
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(FLAGS.peft_model_path+"_full_model", safe_serialization=True)

if __name__ == "__main__":
    app.run(main)