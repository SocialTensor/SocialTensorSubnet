import redis
import json, time


class RedisClient():
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def publish_to_stream(self, stream_name, message):
        message_id = self.client.xadd(stream_name, message)
        print(f"Published message ID: {message_id}")
        return message_id

    def read_from_stream(self, stream_name, count, block):
        messages = self.client.xread({stream_name: '0-0'}, count=count, block=block)  # read $count messages, block for $block miliseconds if no messages
        return messages
    
    def remove_from_stream(self, stream_name, message_id):
        self.client.xdel(stream_name, message_id)
        print(f"Removed message ID: {message_id}")

    def decode_message_stream(self, message_data):
        output = {}
        for key, value in message_data.items():
            output[key.decode('utf-8')] = value.decode('utf-8')
        return output
    
    def clear_stream(self, stream_name):
        self.client.xtrim(stream_name, maxlen=0)
        print(f"Clear stream {stream_name} done !")

    
    async def process_message_from_stream_async(self, stream_name, process_callback, count=10, block=5000, always_ack = False, decode = True, retries = 10):
        while True:
            messages = self.read_from_stream(stream_name, count, block)

            if not messages:
                print("No new messages. Waiting for more...")
                # if retries > 0:
                #     print("No new messages. Waiting for more...")
                #     retries -= 1
                #     time.sleep(5)
                #     continue
                # else:
                #     return

            all_messages = []
            for stream, message_list in messages:
                for message_id, message_data in message_list:
                    if decode:
                        message_data = self.decode_message_stream(message_data)
                    # print(f"Message ID: {message_id}")
                    all_messages.append({
                        "content": message_data,
                        "id": message_id.decode('utf-8')
                    })
                    if always_ack:
                        self.remove_from_stream(stream_name, message_id)
            
            try:
                success_message_ids, error_message_ids  = await process_callback(all_messages)
                if not  always_ack:
                    for message_id in success_message_ids:
                        self.remove_from_stream(stream_name, message_id)
            except Exception as ex:
                print(f"Exception process message in stream: {str(ex)}")