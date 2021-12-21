import numpy as np
import re

def distance(hyp,ref):
	"""Compute Levenshtein Distance
    Args:
		hyp (list): hypothesis list (OCR)
		ref (list):  reference list (Ground Truth)
    Returns:
        Levenshtein Distance (int)
    """
	height = len(ref) + 1
	width = len(hyp) + 1

	dist = np.zeros(height*width, dtype=np.uint8)
	dist = dist.reshape(height,width)

	# Prepare distance array
	for i in range(height):
		for j in range(width):
			if(i==0):
				dist[0][j] = j
			elif(j == 0):
				dist[i][0] = i

	# Compute distance between two vectors
	for i in range(1,height):
		for j in range(1,width):
			if(ref[i-1]!=hyp[j-1]):
				subs = dist[i-1][j-1] + 1
				inse = dist[i][j-1] + 1
				dele = dist[i-1][j]+1
				dist[i][j]=min(subs,inse,dele)
			else:
				dist[i][j] = dist[i-1][j-1]

	return dist[len(ref)][len(hyp)]


def getCer(hyp,ref):

	"""Compute percentage of characters in the reference word
	   that was incorrectly predicted in the OCR output
    Args:
		hyp (string): hypothesis string (OCR)
		ref (string):  reference string (Ground Truth)
    Returns:
        percentage of incorect predicted characters (float)
    """

	assert(isinstance(hyp, str))
	assert(isinstance(ref, str))
	ret=distance(hyp,ref)
	return ret/len(ref)

def getWer(hyp,ref):

	"""Compute percentage of words in the reference text
	   that was incorrectly predicted in the OCR output
    Args:
		hyp (list/string):  hypothesis string/list (OCR)
		ref (list/string):  reference string/list (Ground Truth)
    Returns:
        percentage of incorect predicted words (float)
    """

	if(isinstance(hyp, str)):
		hyp = re.sub(' +', ' ', hyp)
		hyp = hyp.split(' ')
	if(isinstance(ref, str)):
		ref = re.sub(' +', ' ', ref)
		ref = ref.split(' ')

	ret=distance(hyp,ref)
	return ret/len(ref)

if __name__ == "__main__":
	ground_truth, hypothesis = ["my","name", "is", "kenneth"], ["myy", "nime", "iz", "kenneth"]
	ground_truth_s, hypothesis_s = "809475127","80g475Z7"


	mywer = getWer(hypothesis, ground_truth)
	mycer = getCer(hypothesis_s, ground_truth_s)

	mywer_s = getWer(' '.join(hypothesis), ' '.join(ground_truth))
	mycer_s = getCer(' '.join(hypothesis), ' '.join(ground_truth))

	print("WER",mywer)
	print("CER",mycer)

	print("string WER",mywer_s)
	print("string CER",mycer_s)
