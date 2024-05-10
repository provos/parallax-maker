import argparse
import json
import io
from PIL import Image
from urllib import request, parse
import requests

import uuid
import websocket

def load_workflow(workflow_path):
    try:
        with open(workflow_path, 'r') as file:
            workflow = json.load(file)
            return json.dumps(workflow)
    except Exception as e:
        print(f"Failed to load workflow: {e}")
        return None


def patch_inpainting_workflow(
        workflow, image, mask, prompt, negative_prompt,
        strength=0.8, steps=1, cfg_scale=5.0):
    workflow = json.loads(workflow)

    # find the sampler note in workflow
    sampler_id = [key for key, value in workflow.items(
    ) if value['class_type'] == 'KSampler'][0]

    sampler = workflow.get(sampler_id)
    sampler['inputs']['denoise'] = strength
    sampler['inputs']['steps'] = steps
    sampler['inputs']['cfg'] = cfg_scale

    positive_id = sampler['inputs']['positive'][0]
    positive_node = workflow.get(positive_id)
    positive_node['inputs']['text'] = prompt

    negative_id = sampler['inputs']['negative'][0]
    negative_node = workflow.get(negative_id)
    negative_node['inputs']['text'] = negative_prompt

    latent_id = sampler['inputs']['latent_image'][0]
    latent_node = workflow.get(latent_id)

    image_id = latent_node['inputs']['pixels'][0]
    mask_id = latent_node['inputs']['mask'][0]

    image_node = workflow.get(image_id)
    mask_node = workflow.get(mask_id)

    image_node['inputs']['image'] = image
    mask_node['inputs']['image'] = mask

    return workflow


def queue_prompt(server_address, client_id, workflow):
    # send the workflow to the server
    p = {
        "prompt": workflow,
        "client_id": client_id
    }
    data = json.dumps(p).encode('utf-8')
    req = request.Request(f"http://{server_address}/prompt", data=data)
    return json.loads(request.urlopen(req).read())


def upload_image(server_address, file, overwrite=False):
    try:
        files = {
            'image': open(file, 'rb'),
        }
        data = {
            'type': 'input',
            'overwrite': str(overwrite).lower()
        }

        # Making the POST request
        response = requests.post(
            f'http://{server_address}/upload/image', files=files, data=data,
            timeout=500)

        if response.status_code == 200:
            data = response.json()
            path = data['name']
            return path
        else:
            print(f"HTTP Error: {response.status_code} - {response.reason}")
            return None

    except Exception as error:
        print(f"An error occurred: {error}")
        return None


def get_image(server_address, filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = parse.urlencode(data)
    with request.urlopen(f"http://{server_address}/view?{url_values}") as response:
        return response.read()
    
    
def get_images(server_address, client_id, prompt_id):
    ws = websocket.WebSocket()
    ws.connect(f"ws://{server_address}/ws?clientId={client_id}")

    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history(server_address, prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(
                        server_address,
                        image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
            output_images[node_id] = images_output

    return output_images


def get_history(server_address, prompt_id):
    with request.urlopen(f"http://{server_address}/history/{prompt_id}") as response:
        return json.loads(response.read())


def inpainting_comfyui(
        server_address, workflow_path,
        image_path, mask_path,
        prompt, negative_prompt,
        strength=0.5, steps=40, cfg_scale=5.0):
    workflow = load_workflow(workflow_path)
    if workflow is None:        
        return None

    client_id = str(uuid.uuid4())

    image = upload_image(server_address, image_path, overwrite=True)
    mask = upload_image(server_address, mask_path, overwrite=True)

    workflow = patch_inpainting_workflow(
        workflow, image, mask, prompt, negative_prompt,
        strength=strength, steps=steps, cfg_scale=cfg_scale)

    status = queue_prompt(server_address, client_id, workflow)
    prompt_id = status['prompt_id']

    images = get_images(server_address, client_id, prompt_id)

    # XXX - only one image for now
    id = list(images.keys())[0]
    image_data = images[id][0]
    image = Image.open(io.BytesIO(image_data))
    return image


def main():
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument('-w', '--workflow', type=str,
                           help='Path to the workflow file')
    argsparse.add_argument('-i', '--image', type=str,
                           help='Path to the image file')
    argsparse.add_argument('-m', '--mask', type=str,
                           help='Path to the mask file')
    argsparse.add_argument('-p', '--prompt', type=str,
                           default='A beautiful landscape with a mountain in the background',
                           help='Positive prompt')
    argsparse.add_argument('-n', '--negative-prompt', type=str,
                           default='out of focus',
                           help='Negative prompt')
    argsparse.add_argument('-s', '--strength', default=0.8,
                           type=float, help='Strength')
    argsparse.add_argument('-e', '--steps', type=int, default=40,
                           help='Number of steps for the sampler')
    args = argsparse.parse_args()

    server_address = "localhost:8188"

    image = inpainting_comfyui(
        server_address, args.workflow,
        args.image, args.mask, 
        args.prompt, args.negative_prompt, args.strength)
    image.show()


if __name__ == '__main__':
    main()
