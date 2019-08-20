var pathURL = window.location.pathname

var protocolURL = window.location.protocol
var hostURL = window.location.host
var pathArray = window.location.pathname.split('/');
var secondLevelLocation = pathArray[0];
var lastLevelLocation = pathArray[pathArray.length-1];
var oldURL = window.location.protocol + "//" + window.location.host + "/" + window.location.pathname + window.location.search

/*
// Change filename if it is in the URL
if (lastLevelLocation.includes(".html")) {
	pathArray[pathArray.length-1] = 'material.html';
}

var newPathname = "";
for (i = 0; i < pathArray.length; i++) {
  newPathname += "/";
  newPathname += pathArray[i];
}

// var newURL = window.location.protocol + "//" + window.location.host + "/" + newPathname + window.location.search
*/

// course-specific URL
var courseURL = "https://chalmers.instructure.com/courses/7773/";

// Relative link to file if the current filename is in the URL
var newURL = "";
if (pathURL.includes(".html")) {
	newURL = 'material.html';
} else {
	newURL = courseURL + "external_tools/221";
}

var material = "https://chalmers.instructure.com/courses/7773/external_tools/221";
var syllabus = "https://chalmers.instructure.com/courses/7773/assignments/syllabus";
var schedule = "https://chalmers.instructure.com/courses/7773/external_tools/223";
var project1 = "https://chalmers.instructure.com/courses/7773/assignments/4895";
