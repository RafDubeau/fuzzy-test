import os
import sys

# Add the 'local_dependencies' directory to sys.path
current_script_directory = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(current_script_directory, "local_dependencies")
if path not in sys.path:
    sys.path.append(path)

import requests
import json
from colorama import Fore


def get_endpoint_url(endpoint: str) -> str:
    return f"https://{endpoint.replace('_', '-')}-hkvk3lrbva-uc.a.run.app"


def http_call(endpoint: str, method: str, verbose=True, **kwargs):
    if len(kwargs) > 0:
        headers = {"Content-Type": "application/json"}
    else:
        headers = None

    if False and method.upper() == "GET":
        # Not working for some reason. Gives 400 error "You client has issued a malformed or illegal request"
        response = requests.get(
            url=get_endpoint_url(endpoint), json=kwargs, headers=headers
        )
    else:
        response = requests.post(
            url=get_endpoint_url(endpoint), json=kwargs, headers=headers
        )

    good = response.status_code == 200
    colorcode = Fore.GREEN if good else Fore.RED

    try:
        response_data = json.loads(response.text)

        if verbose:
            print(
                f"{Fore.BLUE}{method.upper()}{Fore.RESET} {endpoint} - {colorcode}{response.status_code}{Fore.RESET}:\n{json.dumps(response_data, indent=2)}"
            )
    except:
        response_data = response.text

        if verbose:
            print(
                f"{Fore.BLUE}{method.upper()}{Fore.RESET} {endpoint} - {colorcode}{response.status_code}{Fore.RESET}: {response_data}"
            )

    if not good:
        raise Exception(response_data)

    return response_data


def post(endpoint: str, verbose=True, **kwargs):
    return http_call(endpoint, "POST", verbose, **kwargs)


def get(endpoint: str, verbose=True, **kwargs) -> dict | str:
    return http_call(endpoint, "GET", verbose, **kwargs)


video_groups = [
    [
        "d01dcac8-7623-417c-a043-88bc2ae11729",
        "149a2b22-d08f-407c-a939-c628462408cf",
        "5eed2026-e23d-4fd2-beb8-f348bbfd9587",
        "90821581-9ddc-4e2e-91d5-13af8648de8c",
        "3ae1c5df-a315-4794-a7df-9794ecc0eeb9",
        "d59a5c8f-2167-4626-9811-42f5cfef6fbb",
        "44e01cc2-be47-433b-bf67-af1e1edab0d2",
        "ddbca9d3-2957-4b8c-a1e8-86fcd154a3b4",
        "89785b29-5460-4f56-ae25-bc27dd159cc1",
        "5bb3a982-32fa-4253-9069-e50081546112",
    ],
    [
        "06c159dd-4354-4c6a-8009-b2b391697caa",
        "f96fbec1-2788-4634-9b65-d891747f9ddc",
        "4b322c03-9adc-424a-a6bf-1a07681758da",
        "152df41b-3605-422b-9ff8-05e15ba4c1b3",
        "428cd952-b1d6-4a27-af13-29c80292124c",
        "93ebbae1-b0d0-46c0-b500-fde484533c46",
        "f6f212d1-778b-4e7b-ab42-09655e4c4ede",
        "6582b8bd-372d-4eb3-af91-5dc6c6510143",
        "c4137dfe-5651-4b02-98f8-80fe00f9f7c1",
        "2f3f21f6-642c-4f7d-8dbe-3570ee4dba3f",
    ],
    [
        "812451a3-5a44-4fbf-b1e2-c03ad27daf19",
        "5601f3cd-3471-427d-840e-23c49c71db10",
        "b0e53dac-20a3-4d12-b520-9a63b85d3b10",
        "ddef971d-4c29-4bf6-895d-6d4feb2d0b86",
        "02ffd688-19e3-41ba-8c30-72ec7b99843e",
        "1d29d113-3d96-4151-aad8-b15fa85576bd",
        "58f5dfff-c0ad-4d3a-b728-acd50df2372c",
        "7f711b36-ac61-4551-8f51-f3069c8db7d1",
        "15de192f-a617-4cff-bdc5-8dc8cffbe1f2",
        "023198f2-40b4-4693-b7f1-545584d65072",
    ],
]

all_video_ids = [id for group in video_groups for id in group]


user_groups = [
    [
        "6e5aafc9-83b9-4bdc-b9c5-dc03c20012e4",
        "92bc1766-0fb7-42c7-8867-2614600e9763",
        "c1b5eaf8-34e8-40d5-91ae-f11d83223974",
        "d329c214-bee6-43e8-910b-803b6d38f77f",
        "af16b70c-de2d-4cf6-a144-76a25c7eb079",
        "b6ec1f82-9b66-4085-a818-52112a13e407",
        "ddb75ffc-feb4-4b6b-b8a5-851f4ecc8605",
        "36b16f03-0edd-46f1-9bb8-ab78d4d985a5",
        "484c4a9d-629b-4003-938e-a011c8e0c72c",
        "fc60a42c-49a9-4a61-98a8-6f1c1040fb5e",
        "7570e576-6093-49ba-9980-c59ca4ec63e2",
        "bd7222c2-6a61-4a8f-8b6d-c124c85c554c",
        "58493a9a-c3a5-4b01-af89-61dbf324234c",
        "8fc62ee2-cd5c-49d5-93c5-b8e5de90ee6b",
        "f5ae836a-c8d0-4769-bffa-e2ed0d153997",
        "efb236d2-7781-4915-8ae9-799e682af3d4",
        "90e512ec-dd64-4f9b-a30e-76698d35eb95",
        "94e54f7e-ee47-49aa-8962-a5f7116e06de",
        "14f5cdb0-c575-4f73-9ad7-468027026613",
        "4174dbc2-6d08-471f-a22a-3cc4c1fed698",
        "aec135f6-2137-4535-aabd-4d4f26c12d3a",
        "2f21b77e-7302-4e49-8e19-e39d5ebc3810",
        "55b25e07-ef74-48d4-8222-758570503f60",
        "4eb2f73b-35b0-4180-b2f0-3a101d11ca34",
        "b5274eff-5bba-4c50-890c-0c0011ce8f1a",
    ],
    [
        "a4b1d0cb-2b1f-466a-ba66-bfaf08e1c2f5",
        "f217c0ac-4ab8-47ca-a217-bd513b55592c",
        "6200bf8d-68c5-48f9-be9c-0d373eba5290",
        "c2eccbb9-72d6-4d8d-9c46-c6042853b3af",
        "d619bd75-a94e-42d0-8fbd-42e92283b106",
        "ea8704f3-e2ae-44eb-a19b-2156d7e0d76a",
        "db344bc6-0b4b-4336-ad79-132881b53748",
        "3c7c75a8-c086-4411-bd2a-d8433203de56",
        "328e8d37-9bf0-4baf-9036-38d4e42a3c49",
        "8cc1f427-5679-45a2-9189-747096974e64",
        "538c4638-0995-4b31-a5d2-49628e1a7bb1",
        "78ebb5ce-111b-44c7-b29a-64eeef2f6912",
        "43770b92-94fb-4cd9-b3ee-8fc4bf6a265f",
        "27cb4e27-63ec-4b25-9e14-40efb77c6be1",
        "f453f215-9008-427c-b178-4bdb5e8fe8e8",
        "c73255e5-4518-4738-a5d4-d43bb480e705",
        "da61607c-88e0-4efc-898e-964d638f309e",
        "0eb94371-4231-48ea-9139-1ab4addca30e",
        "7002104e-7336-499a-a71c-e8e7e223bf01",
        "64e7544c-c965-414a-924b-8cfbea8a39cf",
        "3ed044d0-44de-4ccf-9c14-8da7c96e4fa6",
        "bfecf043-fe13-4a94-b598-0c1681176981",
        "daeb324a-21fb-41f7-949b-07e7e03b152e",
        "866f0ef7-c612-4ecb-815c-d420b9781e96",
        "e017a3b8-3271-4b5b-801a-be96e907be6c",
    ],
]

all_user_ids = [id for group in user_groups for id in group]

n_users = len(all_user_ids)
n_videos = len(all_video_ids)
