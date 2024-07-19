
# fixing unicode error in google colab
# import locale

# locale.getpreferredencoding = lambda: "UTF-8"



from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
template = ChatPromptTemplate.from_messages([
    ("system", "You are a good assistant named Typhoon."),
    ("system", "Task : Your task is to perform sentiment analysis on user input."),
    ("system", "Format: Your answer must be in the format of {{'sentiment':[sentiment], 'confident_score':[score]}}."),
    ("system", "Value for [sentiment] must be a string of either 'good', 'bad', or 'neutral' only."),
    ("system", "Value for [score] must be a floating point between 1 and 0. confident score 1 mean the most confident and 0 mean the least confident."),
    ("user", "โจรวิ่งหนีตำรวจลงคลอง"),
    ("assistant", "{{'sentiment' : 'bad', 'confident_score' : 0.95}}"),
    ("user", "ป่าไม้ในประเทศไทย"),
    ("assistant", "{{'sentiment' : 'neutral', 'confident_score' : '0.85'}}"),
    ("user", "ไฟไหม้ตึกร้างบริเวณหลังโรงเรียนอนุบาล"),
    ("assistant", "{{'sentiment' : 'bad', 'confident_score' : 0.99}}"),
    ("user", "ประเทศไทยชนะรางวัลกีฬาแบดมินตัน"),
    ("assistant", "{{'sentiment' : 'good', 'confident_score' : 0.95}}"),
    ("user", "อยากติด #จุฬา ต้องมูที่ไหน✨ . แนะนำที่พระบรมราชานุสาวรีย์สองรัชกาล สถานที่ศักดิ์สิทธิ์ของชาวจุฬา เหมาะกับคนอยากติดจุฬาสุด ๆ"),
    ("assistant", "{{'sentiment' : 'good', 'confident_score' : 0.99}}"),
    ("user", "{user_input}")
])
llm = Ollama(
    base_url = "http://192.168.123.110:11434",
    model="llama3-typhoon-8b:latest",
    temperature=0.1,  # Adjust temperature for diversity
) 
def generate_response(user_input):
    try:
        # Generate the prompt value with context and task
        prompt_value = template.invoke(
            {
                "user_input": user_input
            }
        )
    except Exception as e:
        print(f"Error during template invocation: {e}")
        prompt_value = None

    if prompt_value:
        try:
            # Format the input and get the response from the LLM
            formatted_input = prompt_value['content'] if isinstance(prompt_value, dict) else prompt_value
            response = llm.invoke(formatted_input)
            return response
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            return None
        

user_input = "หลักสูตรแพทย์อินเตอร์ 4 ปี แห่งแรกในไทย"
response = generate_response(user_input)
print(response)