
def out(name, tweets, classifications, score, y, labels):

    correct = 0
    predYES = 0
    predNO = 0
    yYES = 0
    yNO = 0
    TPyes = 0
    TPno = 0

    with open(f'trace_NB-BOW-{name}.txt', 'w') as f:
        for x in range(len(tweets)):
            tweet = list(tweets.keys())[x]
            f.write(f'{tweet}  {classifications[tweet]}  {score[tweet]:e}  {y[tweet]}  {labels[tweet]}\r')

            if labels[tweet] == 'correct': 
                correct += 1 #num of correct classified predicitons
            if classifications[tweet] == 'yes': 
                predYES += 1 #num of yes the model predicted
            else:
                predNO += 1  #num of no the model predicted
            if y[tweet] == 'yes':
                yYES += 1 #num of yes ground truth
            else:
                yNO += 1 #num of no ground truth
            if labels[tweet] == 'correct' and classifications[tweet] == 'yes':
                TPyes += 1 #num of yes model correctly classified
            if labels[tweet] == 'correct' and classifications[tweet] == 'no':
                TPno += 1 #num of no model correctly classified

    acc = correct / len(tweets) #accuracy
    yes_P = TPyes / predYES #precision yes
    no_P = TPno / predNO #precision no
    yes_R = TPyes / yYES #recall yes
    no_R = TPno / yNO #recall no
    f1yes = (2*yes_P*yes_R)/(yes_P+yes_R) #f1 yes
    f1no = (2*no_P*no_R)/(no_P+no_R) #f1 no

    with open(f'eval_NB-BOW-{name}.txt', 'w') as f:
        f.write(f'{acc:.4}\r{yes_P:.4}  {no_P:.4}\r{yes_R:.4}  {no_R:.4}\r{f1yes:.4}  {f1no:.4}')

