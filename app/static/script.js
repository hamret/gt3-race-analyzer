let player;
let telemetryData = [];
let leaderboardData = [];

// This function creates an <iframe> (and YouTube player)
// after the API code downloads.
function onYouTubeIframeAPIReady() {
    player = new YT.Player('player', {
        // Set the height and width to 100% to fill the container
        height: '100%',
        width: '100%',
        // Video ID from the URL we found
        videoId: '71KEr_S-S_c',
        playerVars: {
            'playsinline': 1,
            'autoplay': 0, // Autoplay is disabled
            'controls': 1, // Show player controls
            'start': 15 // Start the video at 15s, where our telemetry begins
        },
        events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange
        }
    });
}

// The API will call this function when the video player is ready.
function onPlayerReady(event) {
    // We can start loading our data now
    loadData();
}

// The API calls this function when the player's state changes.
function onPlayerStateChange(event) {
    if (event.data == YT.PlayerState.PLAYING) {
        // Start the telemetry update loop
        setInterval(updateTelemetry, 100); // Update 10 times per second
    }
}

async function loadData() {
    try {
        const [telemetryRes, leaderboardRes] = await Promise.all([
            fetch('../static/sample_telemetry.json'),
            fetch('../static/sample_leaderboard.json')
        ]);

        telemetryData = await telemetryRes.json();
        leaderboardData = await leaderboardRes.json();

        populateLeaderboard();

    } catch (error) {
        console.error("Error loading data:", error);
    }
}

function populateLeaderboard() {
    const leaderboardBody = document.querySelector('#leaderboard tbody');
    leaderboardBody.innerHTML = ''; // Clear existing data

    leaderboardData.forEach(driver => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${driver.position}</td>
            <td>${driver.driverName}</td>
            <td>${driver.gapToLeader}</td>
        `;
        leaderboardBody.appendChild(row);
    });
}

function updateTelemetry() {
    if (!player || typeof player.getCurrentTime !== 'function') return;

    const currentTime = player.getCurrentTime();

    // Find the closest telemetry snapshot
    const currentData = telemetryData.reduce((prev, curr) => {
        return (Math.abs(curr.timestamp - currentTime) < Math.abs(prev.timestamp - currentTime) ? curr : prev);
    });

    if (currentData) {
        document.getElementById('telemetry-speed').textContent = currentData.speed;
        document.getElementById('telemetry-gear').textContent = currentData.gear;
        document.getElementById('telemetry-rpm').textContent = currentData.rpm;
        
        const throttleBar = document.getElementById('throttle-bar');
        throttleBar.style.width = `${currentData.throttle * 100}%`;

        const brakeBar = document.getElementById('brake-bar');
        brakeBar.style.width = `${currentData.brake * 100}%`;
    }
}
