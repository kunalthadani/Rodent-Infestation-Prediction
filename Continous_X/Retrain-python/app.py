from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/trigger-train', methods=['POST'])
def run_compose():    
    try:
        result_first = subprocess.run(
            ['ansible-playbook', 'playbook.yml'],
            capture_output=True, text=True, check=True
        )
        result = subprocess.run(
            ['ansible-playbook', 'playbook2.yml'],
            capture_output=True, text=True, check=True
        )
        return jsonify({'new_model_version': result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.stderr}), 500


@app.route('/trigger-training', methods=['POST','GET'])
def run_compose_test():
        return jsonify({'new_model_version': '2'})

if __name__ == '__main__':
    app.run(debug=True)
