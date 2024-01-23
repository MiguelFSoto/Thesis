from pypdf import PdfReader

reader = PdfReader("arxivtest.pdf")
number_of_pages = len(reader.pages)
print(number_of_pages)
page = reader.pages[0]
text = page.extract_text()
#print(text)
corpus = ""
for page in reader.pages:
    corpus += page.extract_text()
print(corpus)
