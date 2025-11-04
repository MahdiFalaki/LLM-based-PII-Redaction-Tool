import os
os.environ["LLAMA_CPP_LIB"]=os.path.expanduser(
    "~/Codes/mahdi_codes_folder/axolotl/examples/llama.cpp/build/bin/libllama.so"
)
from llama_cpp import llama_cpp
print("Loaded lib:", getattr(llama_cpp._lib, "_name", "<unknown>"))

