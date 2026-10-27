"""Function words dropped from the keyword side of hybrid search.

They match almost every chunk, and RRF rewards any chunk that makes the keyword list,
so a match on "the" or "და" would push unrelated chunks up. Lowercase, as produced
by the 'simple' text-search config. English follows the common NLTK list; Georgian
covers conjunctions, pronouns, question words and auxiliaries that stand as separate words.
"""

ENGLISH = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "get", "gets", "got", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "may", "me", "might", "more",
    "most", "must", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shall", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves",
}

GEORGIAN = {
    "და", "თუ", "ან", "მაგრამ", "რომ", "რადგან", "როცა", "როდესაც", "ხოლო", "თორემ",
    "რა", "როგორ", "როდის", "სად", "ვინ", "რომელი", "რომელ", "რომელმა", "რამდენი",
    "რამდენად", "რატომ", "რისთვის",
    "მე", "შენ", "ის", "ეს", "ეგ", "ამ", "იმ", "ჩვენ", "თქვენ", "ისინი", "ესენი",
    "ჩემი", "შენი", "მისი", "ამის", "იმის", "ჩვენი", "თქვენი", "მათი",
    "არის", "არ", "ვერ", "ნუ", "კი", "ხომ", "უნდა", "შეიძლება", "იყო", "იქნება",
    "ასე", "ისე", "აქ", "იქ", "ახლა", "უკვე", "ჯერ", "კიდევ", "ძალიან", "ყველა",
    "ზე", "თან", "შემდეგ", "მიერ", "შესახებ", "გარდა",
}

STOPWORDS = sorted(ENGLISH | GEORGIAN)
