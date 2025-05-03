#!/bin/sh

# Duration (in seconds) after which the process will be killed
KILL_DELAY=3

get_screen_dimensions() {
    # Use AppleScript to get the bounds of the main screen
    # Returns a list like "0, 0, 1440, 900"
    screen_bounds=$(osascript -e 'tell application "Finder" to get bounds of window of desktop')

    # Extract individual values using 'awk'
    screen_left=$(echo "$screen_bounds" | awk -F',' '{print $1}' | tr -d ' ')
    screen_top=$(echo "$screen_bounds" | awk -F',' '{print $2}' | tr -d ' ')
    screen_right=$(echo "$screen_bounds" | awk -F',' '{print $3}' | tr -d ' ')
    screen_bottom=$(echo "$screen_bounds" | awk -F',' '{print $4}' | tr -d ' ')

    # Calculate screen width and height
    screen_width=$((screen_right - screen_left))
    screen_height=$((screen_bottom - screen_top))
}

WINDOW_WIDTH=(80*5)
WINDOW_HEIGHT=(40*5)

generate_random_position() {
    # Calculate maximum x and y positions to keep the window within the screen
    max_x=$((screen_width - WINDOW_WIDTH))
    max_y=$((screen_height - WINDOW_HEIGHT))

    # Ensure max_x and max_y are non-negative
    if [ "$max_x" -lt 0 ]; then
        max_x=0
    fi

    if [ "$max_y" -lt 0 ]; then
        max_y=0
    fi

    # Generate random x and y using 'jot'
    random_x=$(jot -r 1 0 "$max_x")
    random_y=$(jot -r 1 0 "$max_y")
}

get_screen_dimensions

while true; do
    dir=$(pwd)

    # if mnist_model.pth exists, break
    if [ -f "$dir/mnist_model.pth" ]; then
        break
    fi

    generate_random_position

    # Use osascript with a here-document for better readability and easier escaping
    osascript <<EOF
    tell application "Terminal"
        do script "cd '$dir'; /Users/mike/CLionProjects/pccl/venv/bin/python '$dir'/mnist_peer.py &
PYTHON_PID=\$!
sleep $KILL_DELAY && (kill \$PYTHON_PID || true) && kill -9 \$\$"
        set position of window 1 to {$random_x, $random_y}
    end tell
EOF
    sleep 0.25
done
