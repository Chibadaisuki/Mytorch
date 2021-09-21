import numpy as np


def GreedySearch(SymbolSets, y_probs):
    """Greedy Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    Returns
    ------
    forward_path: str
                the corresponding compressed symbol sequence i.e. without blanks
                or repeated symbols.

    forward_prob: scalar (float)
                the forward probability of the greedy path
                , forward_prob)

    """
    # Follow the pseudocode from lecture to complete greedy search :-)
    forward_path = []
    b = 0
    forward_prob = 1
    for j in range(len(y_probs[0])):
        for k in range(len(y_probs[0][j])):
            a = y_probs[0][j][k]
            num = 0
            for i in range(len(y_probs)):
                if y_probs[i][j][k] > a:
                    a = y_probs[i][j][k]
                    num = i
            forward_prob = forward_prob*y_probs[num][j][k]
            if num == b:
                break
            b = num
            if num == 0:
                break
            else:
                forward_path.append(SymbolSets[num-1])
    forward_path = str(forward_path).replace("'","")
    forward_path = forward_path.replace(",","")
    forward_path = forward_path.replace(" ","")
    forward_path = forward_path.replace("[","")
    forward_path = forward_path.replace("]","")
    return forward_path, forward_prob


##############################################################################


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    """Beam Search.

    Input
    -----
    SymbolSets: list
                all the symbols (the vocabulary without blank)

    y_probs: (# of symbols + 1, Seq_length, batch_size)
            Your batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size.

    BeamWidth: int
                Width of the beam.

    Return
    ------
    bestPath: str
            the symbol sequence with the best path score (forward probability)

    mergedPathScores: dictionary
                        all the final merged paths with their scores.
bestPath = [] 
    forward_path = []
    mergedPathScores={}
    for row in range(len(y_probs[0])):
        if row == 0:
            for j in range(len(y_probs)):
                mergedPathScores[str(SymbolSets[int(j)-1])]=1*(y_probs[j][row])
            print(mergedPathScores)
            mergedPathScore = sorted(mergedPathScores, key=lambda tup: tup[1], reverse=True)
            print(mergedPathScore)
            mergedPathScores = mergedPathScore[:BeamWidth]
        else:
            for i in mergedPathScores:
                for j in range(len(y_probs)):
                    if (i[0]+str(SymbolSets[j-1])) not in mergedPathScores:
                         mergedPathScores[i[0]+str(SymbolSets[j-1])]=(y_probs[j][row])*i[1]
                    else:
                         mergedPathScores[i[0]+str(SymbolSets[j-1])]+=(y_probs[j][row])*i[1]
        mergedPathScores = sorted(mergedPathScores.items(), key=lambda mergedPathScores:mergedPathScores[1], reverse=True)
        mergedPathScores = mergedPathScores[:BeamWidth]
    print(mergedPathScores)

    """
    # Follow the pseudocode from lecture to complete beam search :-)
    bestPath = []
    forward_path = []
    b=-1
    sequences = [[list(), 1.0]] 

    # walk over each step in sequence
    for row in range(len(y_probs[0])):
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(y_probs)):
                if len(seq)>0:
                    if seq[len(seq)-1] == j:
                        kk = -1
                        for k in range(len(all_candidates)):
                            if str(all_candidates[k][0])==str(seq):
                                kk=k
                            else:
                                continue
                        if kk>-1:
                            all_candidates[kk][1] = all_candidates[kk][1] + score *(y_probs[j][row])
                        else:
                            candidate = [seq, score * (y_probs[j][row])]
                            all_candidates.append(candidate)
                    else:
                        flag =0
                        if len(seq)>1 and seq[len(seq)-1] == 0:
                            seq = seq[0:len(seq)-1]
                            flag = 1
                        kk = -1
                        for k in range(len(all_candidates)):
                            if str(all_candidates[k][0])==str(seq+[j]):
                                kk=k
                            else:
                                continue
                        if kk>-1:
                            all_candidates[kk][1] = all_candidates[kk][1] + score *(y_probs[j][row])
                        else:
                            candidate = [seq+[j], score * (y_probs[j][row])]
                            all_candidates.append(candidate)
                        if flag == 1:
                            seq = seq+[0]
                else:
                    candidate = [seq + [j], score * (y_probs[j][row])]
                    all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse =True)
        # select k best
        sequences = ordered[:BeamWidth]
    mergedPathScores={}
    for k in range(len(ordered)):
        b=-1
        for j in range(len(ordered[k][0])):
            if int(ordered[k][0][j]) == b:
                continue
            b = int(ordered[k][0][j])
            if int(ordered[k][0][j]) == 0:
                continue
            else:
                forward_path.append(SymbolSets[int(ordered[k][0][j])-1])
        forward_path = str(forward_path).replace("'","")
        forward_path = forward_path.replace(",","")
        forward_path = forward_path.replace(" ","")
        forward_path = forward_path.replace("[","")
        forward_path = forward_path.replace("]","")
        bestPath = forward_path
        forward_path = []
        if str(bestPath) not in mergedPathScores:
            mergedPathScores[str(bestPath)]=ordered[k][1]
        else:
            mergedPathScores[str(bestPath)]+=ordered[k][1]
    a = 0
    print(mergedPathScores)
    for i in mergedPathScores:
        if mergedPathScores[i]>a:
            a=mergedPathScores[i]
            c=i
    bestPath=c

    return (bestPath, mergedPathScores)
