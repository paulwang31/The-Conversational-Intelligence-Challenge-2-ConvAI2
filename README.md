# The-Conversational-Intelligence-Challenge-2-ConvAI2
Dialogue Agents: retrieve next best uttereance The topic of the last exercise will be around dialogue system, in particular for non goal-oriented (chit-chat) dialogues. For that we will use the data from the ConvAI challenge. 
For each utterance, you will be given N+1 possible answers of which you have to pick out the correct one. The format of the training will be: utterance TAB correctAnswer TAB distractor1 | distractor2 | ... | distractorN-1, and of testing: utterance TAB possibleAnswer1 | ... | possibleAnswerN+1.

1. Team members: Kewei WANG, Jingyi LI

2. Introduction for user:
This file contain the model train process and model prediction process, for the train process, the model we use is Recall@k means, in order to train, just run as follows:
		$> python NLPex3.py --train --model [pathname] --text [trainData]
The pathname can be adjusted by yourself, but must be with .pickle, like model.pickle, the text parameter should be filled with the path for your own train set, it should be a path like ./convai2_fix_723/train_both_original.txt.
So the train process can be run as the following example:
		$> python NLPex3.py --train --model model.pickle --text ./convai2_fix_723/train_both_original.txt

The test process just run as follows:
		$> python NLPex3.py --test --model [pathname] --text [testData]
The input is the same with train process, for the model part, you should recall the model you save in train process, for example, last time you saved as model.pickle, this time you must recall model.pickle, and the text is the test set path, the result is the output of the answers to your input dialogue.
So the test process can be run as the following example:
		$> python NLPex3.py --test --model model.pickle —text ./convai2_fix_723/valid_both_original.txt
