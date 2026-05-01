const BACKEND_URL = window.BACKEND_URL || "http://127.0.0.1:8000";

// Handle form submission
$("#imageSelectionForm").on("submit", function (e) {
  e.preventDefault();

  let file = $(".item-img")[0].files[0];

  if (!file) {
    alert("Please upload an image!");
    return;
  }

  let formData = new FormData();
  formData.append("file", file);

  $.ajax({
    url: BACKEND_URL + "/predict",
    type: "POST",
    data: formData,
    processData: false,
    contentType: false,

    success: function (response) {
      let result = response.predictions[0];

      $(".output").html(
        "Disease: " + result.label +
        "<br>Confidence: " + (result.probability * 100).toFixed(2) + "%"
      );

      // 🔥 SHOW CHATBOT
      $("#chatSection").show();

      // 🔥 STORE DISEASE
      window.currentDisease = result.label;
    },

    error: function (error) {
      console.error(error);
      alert("Backend error!");
    }
  });
});


// CHATBOT FUNCTION
function sendMessage() {
  let input = document.getElementById("userInput").value;
  if (!input) return;

  // ❗ FIX: check disease exists
  if (!window.currentDisease) {
    alert("Please predict disease first!");
    return;
  }

  let chatBox = document.getElementById("chatBox");

  chatBox.innerHTML += "<p><b>You:</b> " + input + "</p>";

  fetch(BACKEND_URL + "/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      message: input,
      disease: window.currentDisease
    })
  })
  .then(res => res.json())
  .then(data => {
    chatBox.innerHTML += "<p><b>Bot:</b> " + data.reply + "</p>";
    chatBox.scrollTop = chatBox.scrollHeight;
  })
  .catch(err => {
    console.error(err);
    chatBox.innerHTML += "<p><b>Bot:</b> Error connecting to server</p>";
  });

  document.getElementById("userInput").value = "";
}