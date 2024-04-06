import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse

server_address = "82.67.70.191:40892"
client_id = "13c08530-8911-4e38-8489-7cded8eddd9d"

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def load_workflow(workflow_path):
    try:
        with open(workflow_path, 'r') as file:
            workflow = json.load(file)
            return workflow
    except FileNotFoundError:
        print(f"The file {workflow_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"The file {workflow_path} contains invalid JSON.")
        return None

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()

        if isinstance(out, str):

            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images
if __name__ == "__main__":

    with open("generation_models/workflow-json/sticker_maker.json", "r") as file:
        workflow_json = file.read()
  

    workflow = json.loads(workflow_json)
    #set the text prompt for our positive CLIPTextEncode
    workflow["2"]["inputs"]["positive"] = "a dog"
    workflow["4"]["inputs"]["seed"] = 7

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))

    print(ws)
    images = get_images(ws, workflow)


    #Commented out code to display the output images:
    i=0
    for node_id in images:
        for image_data in images[node_id]:
            i+=1
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))
            # image.show()
            print("out")
            image.save(f"Output{i}.webp")
