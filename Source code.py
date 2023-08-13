import streamlit as st
import spacy
from spacy import displacy
import nltk
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summarizer(docx):
    parser=PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summarizer=LexRankSummarizer()
    summary=lex_summarizer(parser.document,3)
    summary_list=[str(sent) for sent in summary]
    result=''.join(summary_list)
    return result
    
def main():
    st.title("Blog Post Optimizer")
    message=st.text_area("Enter your blog::")
    if st.checkbox("Keywords"):
        nlp=spacy.load('en_core_web_sm')
        docx=nlp(message)
        st.markdown(displacy.render(docx,style='ent',jupyter=False),unsafe_allow_html=True)
    if st.checkbox("Sentiment Analysis"):
        st.write("Polarity represents the fact")
        st.write("Subjectivity represents the personal belief")
        if st.button('Analyse'):
            blob=TextBlob(message)
            result_sentiment=blob.sentiment
            st.success(result_sentiment) 
    if st.checkbox("Text Summarization"):
        if st.button('Summarize'):
            st.write("Using Sumy..")
            summary_result=sumy_summarizer(message)
            st.success(summary_result) 

if __name__=='__main__':
    main()