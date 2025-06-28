from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableParallel

# Document -> run parallel chain to generate notes and quiz from it -> merge this two into single document

load_dotenv()

document = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

# define templates
notes_template = PromptTemplate(
    template="Generate short and simple notes from following text \n {text}",
    input_variables=["text"]
)

quiz_template = PromptTemplate(
    template="Generate 5 short quizzes from following text. \n {text}",
    input_variables=["text"]
)

merge_template = PromptTemplate(
    template="Merge the provided notes and quiz into single documents \n notes -> {notes} \n and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

# define models
openai_model = ChatOpenAI()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

hf_model = ChatHuggingFace(llm=llm)

# define parser
parser = StrOutputParser()

# define parallel chain to generate notes and quiz in parallel
parallel_chain = RunnableParallel({
    # these key names has to match the input variables name for respective templates
    "notes": notes_template | hf_model | parser,
    "quiz": quiz_template | openai_model | parser,
})

# define merge chain
merge_chain = merge_template | hf_model | parser

# define final chain using parallel chain and merge chain
chain = parallel_chain | merge_chain

result = chain.invoke({"text": document})
print(result)


# visualize graph
chain.get_graph().print_ascii()