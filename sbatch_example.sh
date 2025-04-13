cd /cs/student/suryagunukula
python example.py

srun --gpus=1 --nodes=1 --time=24:00:00 --cpus-per-task=4 --pty bash
squeue
nvidia-smi
tmux new -s example
tmux ls
tmux a -t example
squeue | grep suryagunukula