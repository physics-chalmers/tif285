// Returns true if the current page is within an iFrame.
function inIframe () {
    try {
        return window.self !== window.top;
    } catch (e) {
        return true;
    }
}

// course-specific URL
var courseURL = "https://chalmers.instructure.com/courses/15029/";

// Course-specific url if the current page is within an iFrame.
// Relative link to file if the current page is the top window
var material = "";
if (inIframe()) {
	material = courseURL + "external_tools/770";
} else {
	material = 'material.html';
}

var syllabus = "";
if (inIframe()) {
	syllabus = courseURL + "assignments/syllabus";
} else {
	syllabus = 'syllabus.html';
}

var schedule = "";
if (inIframe()) {
	schedule = courseURL + "external_tools/771";
} else {
	schedule = 'schedule.html';
}

var gettingstarted = "";
if (inIframe()) {
	gettingstarted = courseURL + "external_tools/772";
} else {
	gettingstarted = 'gettingstarted.html';
}

var remoteteaching = "";
if (inIframe()) {
	remoteteaching = courseURL + "external_tools/601";
} else {
	remoteteaching = 'remoteteaching.html';
}

// var project1 = "https://chalmers.instructure.com/courses/7773/assignments/4895";
