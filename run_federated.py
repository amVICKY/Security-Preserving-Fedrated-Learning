import subprocess
import time
from utils.config import load_config

def main():

    config = load_config()
    num_clients = config["federated"]["num_clients"]
    num_rounds = config["federated"]["num_rounds"]

    for round_num in range(num_rounds):
        print("="*10," Round ","="*10)
        processes = []
        for client_id in range(num_clients):
            process = subprocess.Popen(
                ["python","-m","client.client",str(client_id)]
            )
            processes.append(process)
    
        for process in processes:
            process.wait()

        print(f"Round {round_num+1} completed")
        time.sleep(2)

if __name__ == "__main__":
    main()