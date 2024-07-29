from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="",
)

completion = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(completion.choices[0].message)

a = {
    "id": "cmpl-fffbffa3648f4d92b0e838bf097b7551",
    "object": "text_completion",
    "created": 1721745088,
    "model": "casperhansen/llama-3-70b-instruct-awq",
    "choices": [
        {
            "index": 0,
            "text": "What a fascinating topic! Here's a story about cryptography that I'd love to share with you:\n\n**The Story of Mary, Queen of Scots, and the Cipher that Changed History**\n\nIn the 16th century, Mary, Queen of Scots, was a powerful figure in European politics. She was a Catholic, and her claim to the English throne was a threat to the Protestant Queen Elizabeth I of England. Mary was eventually forced to flee Scotland and seek refuge in England, where she was placed under house arrest.\n\nDespite her captivity, Mary continued to plot against Elizabeth, seeking to overthrow her and take the throne for herself. To communicate with her co-conspirators, Mary turned to cryptography. She used a complex cipher, known as the \"Babington Cipher,\" to encode her messages.\n\nThe Babington Cipher was a sophisticated system, using a combination of substitution and transposition techniques to conceal the meaning of the messages. Mary and her co-conspirators believed it was unbreakable.\n\nHowever, Elizabeth's spymaster, Sir Francis Walsingham, had other plans. He had a team of cryptanalysts, led by a brilliant mathematician named Thomas Phelippes, who were tasked with cracking the cipher.\n\nAfter months of work, Phelippes finally succeeded in breaking the Babington Cipher. He discovered a message from Mary to her co-conspirators, detailing a plot to assassinate Elizabeth and take the throne.\n\nThe consequences were severe. Mary was put on trial, and the decoded messages were used as evidence against her. In 1587, she was found guilty of treason and executed.\n\nThis event marked a turning point in the history of cryptography. It showed that even the most sophisticated ciphers could be broken, and that cryptography was not a foolproof means of secure communication.\n\nThe story of Mary, Queen of Scots, and the Babington Cipher serves as a reminder of the cat-and-mouse game that has been played between cryptographers and cryptanalysts throughout history. It's a game that continues to this day, with new encryption techniques and methods of attack being developed all the time.\n\nI hope you enjoyed this story! Do you have any questions about cryptography or would you like to hear more stories like this?",
            "finish_reason": "stop",
            "stop_reason": 128009,
        }
    ],
    "usage": {"prompt_tokens": 47, "total_tokens": 500, "completion_tokens": 453},
}
