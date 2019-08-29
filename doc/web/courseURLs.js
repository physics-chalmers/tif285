// Returns true if the current page is within an iFrame.
function inIframe () {
    try {
        return window.self !== window.top;
    } catch (e) {
        return true;
    }
}

// course-specific URL
var courseURL = "https://chalmers.instructure.com/courses/7773/";

// Course-specific url if the current page is within an iFrame.
// Relative link to file if the current page is the top window
var material = "";
if (inIframe()) {
	material = courseURL + "external_tools/221";
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
	schedule = courseURL + "external_tools/223";
} else {
	schedule = 'schedule.html';
}

var gettingstarted = "";
if (inIframe()) {
	schedule = courseURL + "external_tools/264";
} else {
	schedule = 'gettingstarted.html';
}

// var project1 = "https://chalmers.instructure.com/courses/7773/assignments/4895";
