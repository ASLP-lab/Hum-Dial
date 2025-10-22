import whisper
from whisper import tokenizer
tokenizer_obj = tokenizer.get_tokenizer(multilingual=True,num_languages=99)
a = tokenizer_obj.decode([10086])
print(a)
a = tokenizer_obj.decode([199])
print(a)
a = tokenizer_obj.decode([198])
print(a)
a = tokenizer_obj.decode([197])
print(a)
#50362
a = tokenizer_obj.decode([50259])
print(a)

for i in range(120,200):
    a = tokenizer_obj.decode([i])
    print(a)