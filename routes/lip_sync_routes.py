from models.lipSyncTasksModal import LipSyncTask
from flask import Blueprint, jsonify , request
from Controllers import lip_sync_controller
from extensions import mongo
bp = Blueprint('lip_sync_task', __name__, url_prefix='/lip_sync_task')

@bp.route("/" , methods=["POST"])
def submitNewTask():
    try:
        data = request.get_json()
        #validate data 
        audio_url  = data.get("audio_url" , None)
        video_url  = data.get("video_url" , None)
        if not audio_url or not video_url:
            raise Exception("Please provide image and video urls")
        
        response  = lip_sync_controller.submit_task(audio_url=audio_url , video_url=video_url)
        if response.get("task_id"):
            return jsonify({"success": True , "task_id" : response.get("task_id") })
        else:
            raise Exception(response.get("error" , "Something went wrong"))
    except Exception as e:
        return jsonify({"error" : str(e)}), 400
    

@bp.route("/status", methods=["GET"])
def getTaskStatus():
    tasks_collection = mongo.db.lip_sync_tasks

    """
    Fetch the current status of a Wav2Lip task using its task_id.
    Example: GET /status?task_id=<your-task-id>
    """
    try:
        task_id = request.args.get("task_id")
        if not task_id:
            raise Exception("Please provide a valid task_id")

        # Fetch task record from MongoDB
        task_data = tasks_collection.find_one({"task_id": task_id})
        if not task_data:
            raise Exception(f"No task found for ID: {task_id}")

        # Convert to model instance
        task = LipSyncTask.from_dict(task_data)

        return jsonify({
            "success": True,
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
            "output_s3_url": task.output_s3_url,
            "error_log": task.error_log,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400