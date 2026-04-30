//var isEnglish = true; // Track the current language state
var isEnglish = true;

function toggleLanguage() {

  var switchButton = document.querySelector(".switch-button");

  if (isEnglish) {

    // Switch to Hindi
    switchButton.innerText = "Switch to English";

    document.querySelector(".homee").innerText = "होम";
    document.querySelector(".aboutee").innerText = "हमारे बारे में";
    document.querySelector(".servicee").innerText = "सेवाएं";
    document.querySelector(".doctorr").innerText = "डॉक्टर";
    document.querySelector(".contactt").innerText = "संपर्क करें";
    document.querySelector(".diaggg").innerText = "अपनी समस्या का निदान करें";

    document.querySelector(".heading-what-is").innerText = "त्वचा संधान क्या है?";

    document.querySelector(".p-what-is").innerText =
      "त्वचा संधान सभी त्वचा संबंधी समस्याओं के लिए एक सरल निदान प्लेटफ़ॉर्म है। अपनी त्वचा की फोटो लें, समस्या चुनें, पिनकोड डालें और परिणाम प्राप्त करें।";

    isEnglish = false;

  } else {

    // Switch back to English
    switchButton.innerText = "हिंदी में देखें";

    document.querySelector(".homee").innerText = "Home";
    document.querySelector(".aboutee").innerText = "About";
    document.querySelector(".servicee").innerText = "Services";
    document.querySelector(".doctorr").innerText = "Doctors";
    document.querySelector(".contactt").innerText = "Contact";
    document.querySelector(".diaggg").innerText = "Diagnose Your Problem";

    document.querySelector(".heading-what-is").innerText = "What is त्वचा संधान?";

    document.querySelector(".p-what-is").innerText =
      "Tvacha Sandhan is a one-stop diagnosis platform for all skin-related issues. Capture your image, select the issue, enter pincode and get results.";

    isEnglish = true;
  }
}
 /*function toggleLanguage() {
  console.log("Click event detected!"); // Debugging line
  var switchButton = document.querySelector(".switch-button");

  if (isEnglish) {
//     // Switch to Hindi
    switchButton.innerText = "Switch to English";
     document.querySelector(".switch-button").innerText = "Switch to english";

     document.querySelector(".homee").innerText = "होम";
     document.querySelector(".aboutee").innerText = "अबाउट";
     document.querySelector(".servicee").innerText = "सेवाएं";
     document.querySelector(".doctorr").innerText = "डॉक्टर्स";
     document.querySelector(".contactt").innerText = "संपर्क करें";
     document.querySelector(".diaggg").innerText = "अपनी समस्या का निदान करें";

     var headingElement = document.querySelector(".heading-what-is");
     headingElement.innerText = "त्वचा संधान क्या है?";*/

    // var pelt=
//     document.querySelector(".p-what-is").innerText =
//       "त्वचा संधान सभी त्वचा संबंधी मुद्दों और त्वचा संबंधी चिंताओं के लिए एक ही स्थान पर निदान है। हमारी प्रक्रिया असाधारण रूप से सरल है, बस अपनी त्वचा की समस्या का एक फोटो लें, संबंधित समस्या चुनें, अपने शहर का पिनकोड दर्ज करें और आप निदान के लिए तैयार हैं।";

//     document.querySelector(".easy-to-access").innerText =
//       "सभी डिवाइस पर उपलब्ध है";
//     document.querySelector(".easy-to-accessp").innerText =
//       "यह वेबएप इस तरह से बनाया गया है कि इसे सभी डिवाइस पर एक्सेस किया जा सकता है";

//     document.querySelector(".fast-diag").innerText = "त्वरित निदान";
//     document.querySelector(".fast-diagp").innerText =
//       "3 क्लिक में विश्वसनीय प्रारंभिक निदान प्राप्त करें";

//     document.querySelector(".sudden").innerText = "व्यापक निगरानी";
//     document.querySelector(".suddenp").innerText =
//       "वेबसाइट विशिष्ट क्षेत्रों में रोग की व्यापकता को ट्रैक और विश्लेषण करने के लिए आपका पोस्टल कोड एकत्र करती है";

//     document.querySelector(".Steps-you").innerText =
//       "निदान के लिए आपको जिन चरणों का पालन करना होगा";
//     document.querySelector(".Steps-youp").innerText =
//       "आपकी त्वचा संबंधी समस्याओं के निदान के लिए नीचे चार आवश्यक चरण बताए गए हैं।";

//     document.querySelector(".title1").innerText =
//       "नेवबार पर अपनी समस्या का निदान करें बटन पर क्लिक करें";
//     document.querySelector(".title2").innerText =
//       "अपनी त्वचा की समस्या का एक फोटो क्लिक करें और इसे तब तक क्रॉप करें जब तक यह अपलोडर बॉक्स में पूरी तरह फिट न हो जाए";
//     document.querySelector(".title3").innerText =
//       "छवि के बगल में रेडियो बटन का चयन करें जो आपकी त्वचा की समस्या से सबसे अच्छी तरह मेल खाता हो!";
//     document.querySelector(".title4").innerText =
//       "टेक्स्ट बॉक्स में अपने शहर का पिनकोड दर्ज करें और सबमिट बटन दबाएं";

//     isEnglish = false;
//   } else {
//     // Switch back to English
//     switchButton.innerText = "हिंदी में देखें";
//     document.querySelector(".homee").innerText = "Home";
//     document.querySelector(".aboutee").innerText = "About";
//     // Add more translations as needed
//     isEnglish = true;
//   }
// }

// function Onclick1() {
//   console.log("Click event detected!"); // Debugging line
//   document.querySelector(".switch-button").innerText = "Switch to english";

//   document.querySelector(".homee").innerText = "होम";
//   document.querySelector(".aboutee").innerText = "अबाउट";
//   document.querySelector(".servicee").innerText = "सेवाएं";
//   document.querySelector(".doctorr").innerText = "डॉक्टर्स";
//   document.querySelector(".contactt").innerText = "संपर्क करें";
//   document.querySelector(".diaggg").innerText = "अपनी समस्या का निदान करें";

//   var headingElement = document.querySelector(".heading-what-is");
//   headingElement.innerText = "त्वचा संधान क्या है?";

//   // var pelt=
//   document.querySelector(".p-what-is").innerText =
//     "त्वचा संधान सभी त्वचा संबंधी मुद्दों और त्वचा संबंधी चिंताओं के लिए एक ही स्थान पर निदान है। हमारी प्रक्रिया असाधारण रूप से सरल है, बस अपनी त्वचा की समस्या का एक फोटो लें, संबंधित समस्या चुनें, अपने शहर का पिनकोड दर्ज करें और आप निदान के लिए तैयार हैं।";

//   document.querySelector(".easy-to-access").innerText =
//     "सभी डिवाइस पर उपलब्ध है";
//   document.querySelector(".easy-to-accessp").innerText =
//     "यह वेबएप इस तरह से बनाया गया है कि इसे सभी डिवाइस पर एक्सेस किया जा सकता है";

//   document.querySelector(".fast-diag").innerText = "त्वरित निदान";
//   document.querySelector(".fast-diagp").innerText =
//     "3 क्लिक में विश्वसनीय प्रारंभिक निदान प्राप्त करें";

//   document.querySelector(".sudden").innerText = "व्यापक निगरानी";
//   document.querySelector(".suddenp").innerText =
//     "वेबसाइट विशिष्ट क्षेत्रों में रोग की व्यापकता को ट्रैक और विश्लेषण करने के लिए आपका पोस्टल कोड एकत्र करती है";

//   document.querySelector(".Steps-you").innerText =
//     "निदान के लिए आपको जिन चरणों का पालन करना होगा";
//   document.querySelector(".Steps-youp").innerText =
//     "आपकी त्वचा संबंधी समस्याओं के निदान के लिए नीचे चार आवश्यक चरण बताए गए हैं।";

//   document.querySelector(".title1").innerText =
//     "नेवबार पर अपनी समस्या का निदान करें बटन पर क्लिक करें";
//   document.querySelector(".title2").innerText =
//     "अपनी त्वचा की समस्या का एक फोटो क्लिक करें और इसे तब तक क्रॉप करें जब तक यह अपलोडर बॉक्स में पूरी तरह फिट न हो जाए";
//   document.querySelector(".title3").innerText =
//     "छवि के बगल में रेडियो बटन का चयन करें जो आपकी त्वचा की समस्या से सबसे अच्छी तरह मेल खाता हो!";
//   document.querySelector(".title4").innerText =
//     "टेक्स्ट बॉक्स में अपने शहर का पिनकोड दर्ज करें और सबमिट बटन दबाएं";
// }
// Function to find hospitals by pincode
document.addEventListener("DOMContentLoaded", function () {

  const btn = document.getElementById("findHospitalBtn");

  if (!btn) {
    console.log("Button not found");
    return;
  }

  btn.addEventListener("click", function (e) {

    e.preventDefault(); // stop form submit

    const pincodeInput = document.getElementById("pincode");

    if (!pincodeInput) {
      alert("Pincode input not found");
      return;
    }

    const pincode = pincodeInput.value.trim();

    if (!/^\d{6}$/.test(pincode)) {
      alert("Enter valid 6 digit pincode");
      return;
    }

    const url = "https://www.google.com/maps/search/hospitals+near+" + pincode;

    window.open(url, "_blank");

  });

});
/*function findHospitalsByPincode(pincode) {
  // Mock data for demonstration
  const mockData = {
      "110001": [
          { name: "AIIMS Hospital", address: "Ansari Nagar, New Delhi" },
          { name: "Apollo Hospital", address: "Mathura Road, New Delhi" }
      ],
      "560001": [
          { name: "Fortis Hospital", address: "Bannerghatta Road, Bengaluru" },
          { name: "Manipal Hospital", address: "Old Airport Road, Bengaluru" }
      ]
  };

  const hospitals = mockData[pincode];
  if (hospitals) {
      let output = "<h4>Nearest Hospitals:</h4><ul>";
      hospitals.forEach(hospital => {
          output += `<li>${hospital.name} - ${hospital.address}</li>`;
      });
      output += "</ul>";
      document.getElementById("hospitalInfo").innerHTML = output;
  } else {
      document.getElementById("hospitalInfo").innerHTML = "No hospitals found for the entered pincode.";
  }
}

// Event listener for button click
document.getElementById("findHospitalBtn").addEventListener("click", () => {
  const pincode = document.getElementById("pincode").value.trim();

  // Validate pincode
  if (pincode.length === 6 && /^\d+$/.test(pincode)) {
      findHospitalsByPincode(pincode);
  } else {
      document.getElementById("hospitalInfo").innerHTML = "Please enter a valid 6-digit pincode.";
  }
});*/
document.addEventListener("DOMContentLoaded", function () {

  const pharmacyBtn = document.getElementById("findPharmacyBtn");

  if (!pharmacyBtn) {
    console.log("Pharmacy button not found");
    return;
  }

  pharmacyBtn.addEventListener("click", function () {

    const pincode = document.getElementById("pharmacyPincode").value.trim();

    if (!/^\d{6}$/.test(pincode)) {
      alert("Enter valid 6 digit pincode");
      return;
    }

    const url = "https://www.google.com/maps/search/government+medical+store+near+" + pincode;

    window.open(url, "_blank");

  });

});
document.addEventListener("DOMContentLoaded", function () {

  const dermBtn = document.getElementById("findDermBtn");

  if (dermBtn) {
    dermBtn.addEventListener("click", function () {

      const pincode = document.getElementById("dermaPincode").value.trim();

      if (!/^\d{6}$/.test(pincode)) {
        alert("Enter valid 6 digit pincode");
        return;
      }

      const url = "https://www.google.com/maps/search/dermatologist+near+" + pincode;

      window.open(url, "_blank");

    });
  }

});



(function () {
  "use strict";

  /**
   * Easy selector helper function
   */
  const select = (el, all = false) => {
    el = el.trim();
    if (all) {
      return [...document.querySelectorAll(el)];
    } else {
      return document.querySelector(el);
    }
  };

  /**
   * Easy event listener function
   */
  const on = (type, el, listener, all = false) => {
    let selectEl = select(el, all);
    if (selectEl) {
      if (all) {
        selectEl.forEach((e) => e.addEventListener(type, listener));
      } else {
        selectEl.addEventListener(type, listener);
      }
    }
  };

  /**
   * Easy on scroll event listener
   */
  const onscroll = (el, listener) => {
    el.addEventListener("scroll", listener);
  };

  /**
   * Navbar links active state on scroll
   */
  let navbarlinks = select("#navbar .scrollto", true);
  const navbarlinksActive = () => {
    let position = window.scrollY + 200;
    navbarlinks.forEach((navbarlink) => {
      if (!navbarlink.hash) return;
      let section = select(navbarlink.hash);
      if (!section) return;
      if (
        position >= section.offsetTop &&
        position <= section.offsetTop + section.offsetHeight
      ) {
        navbarlink.classList.add("active");
      } else {
        navbarlink.classList.remove("active");
      }
    });
  };
  window.addEventListener("load", navbarlinksActive);
  onscroll(document, navbarlinksActive);

  /**
   * Scrolls to an element with header offset
   */
  const scrollto = (el) => {
    let header = select("#header");
    let offset = header.offsetHeight;

    let elementPos = select(el).offsetTop;
    window.scrollTo({
      top: elementPos - offset,
      behavior: "smooth",
    });
  };

  /**
   * Toggle .header-scrolled class to #header when page is scrolled
   */
  let selectHeader = select("#header");
  let selectTopbar = select("#topbar");
  if (selectHeader) {
    const headerScrolled = () => {
      if (window.scrollY > 100) {
        selectHeader.classList.add("header-scrolled");
        if (selectTopbar) {
          selectTopbar.classList.add("topbar-scrolled");
        }
      } else {
        selectHeader.classList.remove("header-scrolled");
        if (selectTopbar) {
          selectTopbar.classList.remove("topbar-scrolled");
        }
      }
    };
    window.addEventListener("load", headerScrolled);
    onscroll(document, headerScrolled);
  }

  /**
   * Back to top button
   */
  let backtotop = select(".back-to-top");
  if (backtotop) {
    const toggleBacktotop = () => {
      if (window.scrollY > 100) {
        backtotop.classList.add("active");
      } else {
        backtotop.classList.remove("active");
      }
    };
    window.addEventListener("load", toggleBacktotop);
    onscroll(document, toggleBacktotop);
  }

  /**
   * Mobile nav toggle
   */
  on("click", ".mobile-nav-toggle", function (e) {
    select("#navbar").classList.toggle("navbar-mobile");
    this.classList.toggle("bi-list");
    this.classList.toggle("bi-x");
  });

  /**
   * Mobile nav dropdowns activate
   */
  on(
    "click",
    ".navbar .dropdown > a",
    function (e) {
      if (select("#navbar").classList.contains("navbar-mobile")) {
        e.preventDefault();
        this.nextElementSibling.classList.toggle("dropdown-active");
      }
    },
    true
  );

  /**
   * Scrool with ofset on links with a class name .scrollto
   */
  on(
    "click",
    ".scrollto",
    function (e) {
      if (select(this.hash)) {
        e.preventDefault();

        let navbar = select("#navbar");
        if (navbar.classList.contains("navbar-mobile")) {
          navbar.classList.remove("navbar-mobile");
          let navbarToggle = select(".mobile-nav-toggle");
          navbarToggle.classList.toggle("bi-list");
          navbarToggle.classList.toggle("bi-x");
        }
        scrollto(this.hash);
      }
    },
    true
  );

  /**
   * Scroll with ofset on page load with hash links in the url
   */
  window.addEventListener("load", () => {
    if (window.location.hash) {
      if (select(window.location.hash)) {
        scrollto(window.location.hash);
      }
    }
  });

  /**
   * Preloader
   */
  let preloader = select("#preloader");
  if (preloader) {
    window.addEventListener("load", () => {
      preloader.remove();
    });
  }

  /**
   * Initiate glightbox
   */
  const glightbox = GLightbox({
    selector: ".glightbox",
  });

  // /**
  //  * Initiate Gallery Lightbox
  //  */
  const galelryLightbox = GLightbox({
    selector: ".galelry-lightbox",
  });

  /**
   * Testimonials slider
   */
  new Swiper(".testimonials-slider", {
    speed: 600,
    loop: true,
    autoplay: {
      delay: 5000,
      disableOnInteraction: false,
    },
    slidesPerView: "auto",
    pagination: {
      el: ".swiper-pagination",
      type: "bullets",
      clickable: true,
    },
    breakpoints: {
      320: {
        slidesPerView: 1,
        spaceBetween: 20,
      },

      1200: {
        slidesPerView: 2,
        spaceBetween: 20,
      },
    },
  });

  // /**
  //  * Initiate Pure Counter
  //  */
  new PureCounter();
})();
// Do's and Don'ts Redirect
document.addEventListener("DOMContentLoaded", function () {

  const dosLink = document.getElementById("dosDontsLink");

  if (!dosLink) {
    console.log("Do's & Don'ts link not found");
    return;
  }

  dosLink.addEventListener("click", function (e) {

    e.preventDefault();

    window.open(
      "https://www.mayoclinic.org/diseases-conditions/skin-conditions",
      "_blank"
    );

  });

});
// Ayurveda Redirect
document.addEventListener("DOMContentLoaded", function () {

  const ayuLink = document.getElementById("ayurvedaLink");

  if (!ayuLink) {
    console.log("Ayurveda link not found");
    return;
  }

  ayuLink.addEventListener("click", function (e) {

    e.preventDefault();

    window.open(
      "https://www.nhp.gov.in/ayurveda",
      "_blank"
    );

  });

});
