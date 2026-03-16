from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("AI Resume Screening System")

job_description = input("Enter Job Description: ")
resume_text = input("Enter Resume Skills: ")

documents = [job_description, resume_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("Match Score:", round(score,2), "%")

if score > 50:
    print("Candidate Shortlisted")
else:
    print("Candidate Rejected")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("AI Resume Screening System")

job_description = input("Enter Job Description: ")
resume_text = input("Enter Resume Skills: ")

documents = [job_description, resume_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("Match Score:", round(score,2), "%")

if score > 50:
    print("Candidate Shortlisted")
else:
    print("Candidate Rejected")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("AI Resume Screening System")

job_description = input("Enter Job Description: ")
resume_text = input("Enter Resume Skills: ")

documents = [job_description, resume_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("Match Score:", round(score,2), "%")

if score > 50:
    print("Candidate Shortlisted")
else:
    print("Candidate Rejected")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("AI Resume Screening System")

job_description = input("Enter Job Description: ")
resume_text = input("Enter Resume Skills: ")

documents = [job_description, resume_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("Match Score:", round(score,2), "%")

if score > 50:
    print("Candidate Shortlisted")
else:
    print("Candidate Rejected")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("AI Resume Screening System")

job_description = input("Enter Job Description: ")
resume_text = input("Enter Resume Skills: ")

documents = [job_description, resume_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("Match Score:", round(score,2), "%")

if score > 50:
    print("Candidate Shortlisted")
else:
    print("Candidate Rejected")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("AI Resume Screening System")

job_description = input("Enter Job Description: ")
resume_text = input("Enter Resume Skills: ")

documents = [job_description, resume_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("Match Score:", round(score,2), "%")

if score > 50:
    print("Candidate Shortlisted")
else:
    print("Candidate Rejected")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("AI Resume Screening System")

job_description = input("Enter Job Description: ")
resume_text = input("Enter Resume Skills: ")

documents = [job_description, resume_text]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity[0][0] * 100

print("Match Score:", round(score,2), "%")

if score > 50:
    print("Candidate Shortlisted")
else:
    print("Candidate Rejected")