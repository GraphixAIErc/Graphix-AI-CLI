import pandas as pd
import os
import tarfile
import requests
import numpy as np
from graphix.ai.models.model import load_model_and_tokenizer
from graphix.ai.data.data_loader import load_and_preprocess_dataset
from graphix.ai.training.trainer import train_model
from graphix.ai.data.cache import upload_file, setup_local_cache
from graphix.initiate_socket.initiate import sio
from graphix.config.settings import SERVER_URL


def fine_tune(task):
    # Load data
    taskId = task.get("id", {})
    aiModel = task.get("aiModel", {})
    model_type = aiModel.get("type")

    training_data = task.get("trainingData", {})
    training_data_url = training_data.get("trainingDataUrl")
    validation_data_url = training_data.get("validationDataUrl")
    training_dataset_url = training_data_url.get("data")
    validation_dataset_url = validation_data_url.get("data")

    training_parameters = task.get("trainingParameters", {})
    epochs = training_parameters.get("numEpochs")
    batch_size = training_parameters.get("batchSize")
    lr = training_parameters.get("learningRate")

    try:
        results_dir = os.path.join(os.path.dirname(
            __file__), "..", "..", "..", "results")
        save_dir = os.path.join(results_dir, "saved_model")

        if training_dataset_url is None:
            raise Exception(f"trainingDataUrl is required.")

        if validation_dataset_url is None:
            raise Exception(f"validationDataUrl is required.")

        # Load model and tokenizer based on task type
        model, tokenizer = load_model_and_tokenizer(model_type)

        sio.emit('BMAIN: logs', {
            'taskId': taskId,
            'message': 'Model and tokenizer cache setup and loaded complete'
        })

        # Load and preprocess dataset
        train_dataset, eval_dataset = load_and_preprocess_dataset(
            model_type, tokenizer, training_dataset_url, validation_dataset_url)

        sio.emit('BMAIN: logs', {
            'taskId': taskId,
            'message': 'Dataset loaded and ready for processing'
        })

        sio.emit('BMAIN: logs', {
            'taskId': taskId,
            'message': 'Trainer initialized, starting training'
        })

        # Train the model
        train_model(model_type, model, tokenizer, train_dataset,
                    eval_dataset, epochs, batch_size, lr, save_dir)

        sio.emit('BMAIN: logs', {
            'taskId': taskId,
            'message': 'Training completed'
        })

        # # Compress the saved model and tokenizer into a folder
        compressed_file = results_dir + ".tar.gz"
        # with tarfile.open(compressed_file, "w:gz") as tar:
        #     tar.add(results_dir, arcname=os.path.basename(results_dir))

        sio.emit('BMAIN: logs', {
            'taskId': taskId,
            'message': 'Model and tokenizer compressed'
        })

        response = upload_file(
            server_url=SERVER_URL + "/api/file-upload/upload",
            file_path=compressed_file,
        )
        
        sio.emit('BMAIN: logs', {
            'taskId': taskId,
            'message': 'Model and tokenizer uploaded'
        })

        sio.emit('BMAIN: task_update', {
            'taskId': taskId,
            "cid": response['data']['cid'],
            'message': 'Task completed'
        })

    except Exception as e:
        sio.emit('BMAIN: execute_error', {
            'message': str(e),
            'taskId': taskId,
        })
        sio.emit('error', {
            'message': str(e),
            'taskId': taskId,
        })
        print(f"An error occurred during task execution: {str(e)}")


task = {'id': 'cm3idv11a0002uswzgt4429uv', 'title': 'GPT2', 'description': 'This model can be fine-tuned using custom datasets to optimize its performance for specific tasks and achieve better accuracy in targeted applications.', 'taskType': 'FINE_TUNE', 'status': 'PENDING', 'userId': 'cm2a02jjs0000zzug21e5caxs', 'aiModelid': 'cm3h1vrgc0000gmvvo9ex6hor', 'trainingData': {'trainingDataUrl': {'sucess': 'Upload successfully', 'data': {'cid': 'bafkreibj5nno2rgnhej2kiqeg6mmk5lkocs5dtvc6jll5tdl6evs4istze'}}, 'validationDataUrl': {'sucess': 'Upload successfully', 'data': {'cid': 'bafkreieudivn5uy6ii4tzpqv2ozxoiohset57jjoz3cbwmgt3762z6hf6u'}}}, 'trainingParameters': {'learningRate': 0.0001, 'batchSize': 16, 'numEpochs': 3, 'optimizer':
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            'AdamW', 'lossFunction': 'CrossEntropyLoss'}, 'nodeId': None, 'logs': [], 'createdAt': '2024-11-15T06:53:50.975Z', 'updatedAt': '2024-11-15T06:53:50.975Z', 'aiModel': {'id': 'cm3h1vrgc0000gmvvo9ex6hor', 'modelName': 'GPT2', 'type': 'gpt2', 'url': '', 'configUrl': '', 'otherUrl': {}, 'framework': 'Pytorch', 'version': '1.0', 'userId': 'Admin', 'createdAt': '2024-11-14T08:30:43.642Z', 'updatedAt': '2024-11-14T08:30:43.642Z'}}
# fine_tune(task)
