import numpy as np
import re
# https://colab.research.google.com/drive/1IPpwx4rX32rqHKpLz7dc8sOKspUa-YKO?undefined#scrollTo=RVJs4Bk8FjjO

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer


'''
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
'''