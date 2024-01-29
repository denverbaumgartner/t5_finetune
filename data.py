import os
import datasets

class SynthData:

    def __init__(self, dataset='semiotic/spider_dataset_tuning'): 
        self.data = datasets.load_dataset(dataset)
        #self.data = datasets.load_from_disk('data/spider_tuning_dict')

    def format_request(self, question, schema): 
        input_template = "Schema: {schema} translate English to SQL: {question} </s>"
        return input_template.format(schema=schema, question=question)
    
    def get_subset(self, subset='train', type='synthetic_joint'): 
        return self.data[subset].filter(lambda x: x['type'] == type)
    
    def prepare_sources(self, subset='train', type='synthetic_joint', data_dir='data'):
        # Ensure data_dir exists
        os.makedirs(data_dir, exist_ok=True)

        # Create or clear existing files
        open(f'{data_dir}/{subset}.source', 'w').close()
        open(f'{data_dir}/{subset}.target', 'w').close()

        # Get subset data
        subset_data = self.get_subset(subset=subset, type=type)

        # Append data to the files
        for i, row in enumerate(subset_data):
            with open(f'{data_dir}/{subset}.source', 'a') as f_source, \
                open(f'{data_dir}/{subset}.target', 'a') as f_target:
                f_source.write(self.format_request(row['question'], row['schema']) + '\n')
                f_target.write(row['query'] + ' </s>\n')

    def get_unk_tokens(self, tokenizer, value): 
        unk_tokens = []
        tokens = set(tokenizer.tokenize(value))
        for token in tokens: 
            if tokenizer.convert_tokens_to_ids(token) == tokenizer.unk_token_id: 
                unk_tokens.append(token)
        return unk_tokens

    def get_dataset_unk_tokens(self, tokenizer, dataset): 
        unk_tokens = []
        for i, row in enumerate(dataset): 
            text = self.format_request(row['question'], row['schema']) + ' ' + row['query']
            unks = self.get_unk_tokens(tokenizer, text)
            unk_tokens.extend(unks)
        return list(set(unk_tokens))

    def get_all_unk_tokens(self, tokenizer): 
        unk_tokens = []
        unk_tokens.extend(self.get_dataset_unk_tokens(tokenizer, self.data['train']))
        unk_tokens.extend(self.get_dataset_unk_tokens(tokenizer, self.data['val']))
        unk_tokens.extend(self.get_dataset_unk_tokens(tokenizer, self.data['test']))
        return list(set(unk_tokens))
        