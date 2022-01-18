The code used to train the different model explained in the report is divided
into the following 3 folders:
- Task_A that contains the code relative to the task Task_A
- Task_B that contains the code relative to the task Task B
- NewDataset that containse the code for training and testing the models over the new dataset (As Extra)

In Task_A you will find
- Task_A_Baseline.ipynb that contains the code for the BiLSTM based model with binary classification of each term
- Task_A_Baseline_3_classes.ipynb that contains the code for the BiLSTM based model with multiclass classification of each term

The other file are based on BERT and there are types of file
- first_word_piece_combination are the file that contains this modality for dealing with terms with multiple tokens
- _n_hidden_layer are the files relative to the model that takes the last n hidden layers from the bert output
- _3_classes are the files containing the code for the models that use a multi class classification

In Task_B you will find
- Task_B_bert_without_target_term_max_avg_combinations.ipynb containing the code for the model that uses max or avg combination techniques (in order to combine word pieces of the same term)
- Task_B_special_token.ipynb containing the code for the model using the special token
- both these model take the last four hidden layers of Bert 

In NewDataset you will find
- New_Dataset_Task_A_sum_max_mean_combination_4_hidden_layers.ipynb that is the model with the three combination techniques over the new dataset for task A
- New_Dataset_Task_B_special_token_4_hidden_layers.ipynb that is the model with the special token new dataset for task B

