import sys
import matplotlib.pyplot as plt

# run sed -n '/Training Epoch:/p' *.log > losses.txt beforehand
# run sed -n '/losses_local_local/p' *.log > local_losses.txt beforehand
# run sed -n '/losses_global_local/p' *.log > local_losses.txt beforehand
# run sed -n '/losses_ic2/p' *.log > local_losses.txt beforehand
# run sed -n '/losses_ic1/p' *.log > local_losses.txt beforehand

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python plot_losses.py LOSSES_FILENAME")

  losses_file = sys.argv[1]
  f = open(losses_file, "r")

  epoch_losses = f.read().splitlines()
  epochs = []
  losses = []
  for line in epoch_losses:
    e_no, e_loss = line.split(',')
    e_no = e_no.rsplit(' ', 1)[-1]
    e_loss = e_loss.rsplit(' ', 1)[-1]

    epochs.append(int(e_no))
    losses.append(float(e_loss))
  
  plt.plot(epochs, losses)
  plt.ylabel('Losses')
  plt.xlabel('Epochs')
  plt.show()
