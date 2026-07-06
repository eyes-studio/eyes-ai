import json

class Tokenizer:
    def __init__(self, filename):
        self.filnam = filename

    def Tokanizate(self, text):
        with open(self.filnam, 'r', encoding='utf-8') as file:
            jsonlist = json.load(file)
        
        text = text.lower()
        inplist = text.split()

        token = []

        for texfo in inplist:
            if texfo in jsonlist:
                token.append(jsonlist[texfo])
            else:
                last_tok = max(jsonlist.values()) if jsonlist else 0
                new_token_num = last_tok + 1
                
                jsonlist[texfo] = new_token_num
                token.append(new_token_num) 
        
        with open(self.filnam, 'w', encoding='utf-8') as file:
            json.dump(jsonlist, file, ensure_ascii=False, indent=2)

        return token

    def get_tokens(self, text):
        tokens = self.Tokanizate(text) 
    
        if not tokens:
            return []
    
        max_token = max(tokens)
        normalized = [t / max_token for t in tokens]

        return normalized
