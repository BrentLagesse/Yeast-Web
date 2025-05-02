from django.shortcuts import get_object_or_404, get_list_or_404, redirect
from django.http import JsonResponse
from django.template.response import TemplateResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from core.models import DVLayerTifPreview, UploadedImage
from core.mrcnn.my_inference import predict_images
from core.mrcnn.preprocess_images import preprocess_images
from .utils import tif_to_jpg
from core.dv_channel_parser import extract_channel_config

from yeastweb.settings import MEDIA_ROOT
from pathlib import Path
import json


def pre_process_step(request, uuids):
    """
    GET: Render previews + sidebar (with auto-detected channel order).
    POST: Run preprocess + inference on every UUID, then redirect.
    """
    uuid_list = uuids.split(',')
    total_files = len(uuid_list)

    # clamp file_index into [0, total_files-1]
    current_file_index = int(request.GET.get('file_index', 0))
    current_file_index = max(0, min(current_file_index, total_files - 1))

    # build sidebar list, including the 4-channel order per file
    file_list = []
    for uid in uuid_list:
        uploaded = get_object_or_404(UploadedImage, uuid=uid)

        # try reading existing channel_config.json
        cfg_path = Path(MEDIA_ROOT) / uid / 'channel_config.json'
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            detected_channels = [ch for ch, _ in sorted(cfg.items(), key=lambda t: t[1])]
        else:
            # fallback: parse header of first .dv file
            dv_files = list((Path(MEDIA_ROOT) / uid).glob('*.dv'))
            if dv_files:
                cfg = extract_channel_config(str(dv_files[0]))
                detected_channels = [ch for ch, _ in sorted(cfg.items(), key=lambda t: t[1])]
            else:
                detected_channels = []

        file_list.append({
            'uuid': uid,
            'name': uploaded.name,
            'detected_channels': detected_channels,
        })

    # current file previews
    current_uuid = uuid_list[current_file_index]
    uploaded_image = get_object_or_404(UploadedImage, uuid=current_uuid)
    preview_images = get_list_or_404(DVLayerTifPreview, uploaded_image_uuid=current_uuid)

    # POST: preprocess + predict all, then redirect
    if request.method == "POST":
        for image_uuid in uuid_list:
            img_obj = get_object_or_404(UploadedImage, uuid=image_uuid)
            out_dir = Path(MEDIA_ROOT) / image_uuid

            prep_path, prep_list = preprocess_images(image_uuid, img_obj, out_dir)
            tif_to_jpg(Path(prep_path), out_dir)
            predict_images(prep_path, prep_list, out_dir)

        return redirect(f'/image/{uuids}/convert/')

    # AJAX navigation
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({
            'images': [
                {'file_location': {'url': img.file_location.url}}
                for img in preview_images
            ],
            'file_name': uploaded_image.name,
            'current_file_index': current_file_index,
        })

    # Normal render
    return TemplateResponse(request, "pre-process.html", {
        'images': preview_images,
        'file_name': uploaded_image.name,
        'current_file_index': current_file_index,
        'total_files': total_files,
        'uuids': uuids,
        'file_list': file_list,
    })


@require_POST
@csrf_exempt
def update_channel_order(request, uuid):
    """
    POST {order: ["DIC","DAPI","mCherry","GFP"]}
    → overwrite channel_config.json in MEDIA_ROOT/<uuid>/
    """
    try:
        data = json.loads(request.body)
        new_order = data.get('order', [])
        expected = {"mCherry", "GFP", "DAPI", "DIC"}
        if set(new_order) != expected:
            return JsonResponse({'error': 'invalid channel list'}, status=400)

        # new: 0–3 mapping to match your layer filenames
        mapping = {ch: i for i, ch in enumerate(new_order)}


        cfg_path = Path(MEDIA_ROOT) / uuid / 'channel_config.json'
        if not cfg_path.exists():
            return JsonResponse({'error': 'config not found'}, status=404)

        # SAVE: overwrite the JSON file with new mapping
        cfg_path.write_text(json.dumps(mapping))
        return JsonResponse({'status': 'ok'})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
