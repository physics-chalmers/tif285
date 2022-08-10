// Returns true if the current page is within an iFrame.
function inIframe () {
    try {
        return window.self !== window.top;
    } catch (e) {
        return true;
    }
}

// course-specific URL
var bookURL = "https://cforssen.gitlab.io/tif285-book/";
var bookIntroURL = bookURL + "content/Intro/";
var bookLinRegURL = bookURL + "content/LinearRegression/";
var bookBayesStatURL = bookURL + "content/BayesianStatistics/";
var bookMLURL = bookURL + "content/MachineLearning/";
var gitURL = "https://github.com/physics-chalmers/tif285/";
var gitIntroURL = gitURL + "blob/master/doc/LectureNotes/content/Intro/";
var gitLinRegURL = gitURL + "blob/master/doc/LectureNotes/content/LinearRegression/";
var gitBayesStatURL = gitURL + "blob/master/doc/LectureNotes/content/BayesianStatistics/";
var gitMLURL = gitURL + "blob/master/doc/LectureNotes/content/MachineLearning/";

// course-specific URL
var courseURL = "https://chalmers.instructure.com/courses/20232/";

// Course-specific url if the current page is within an iFrame.
// Relative link to file if the current page is the top window
var material = "";
if (inIframe()) {
	material = courseURL + "external_tools/958";
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
	schedule = courseURL + "external_tools/960";
} else {
	schedule = 'schedule.html';
}

var gettingstarted = "";
if (inIframe()) {
	gettingstarted = courseURL + "external_tools/957";
} else {
	gettingstarted = 'gettingstarted.html';
}

var remoteteaching = "";
if (inIframe()) {
	remoteteaching = courseURL + "external_tools/959";
} else {
	remoteteaching = 'remoteteaching.html';
}

var discussions = "";
if (inIframe()) {
	discussions = courseURL + "external_tools/962";
} else {
	discussions = 'https://app.yata.se/course/17820382-e29e-479f-8e4e-15ea2618d2c7/posts';
}

// var project1 = "https://chalmers.instructure.com/courses/7773/assignments/4895";

// course PM on student portals
var coursePMchalmers = "https://www.student.chalmers.se/sp/course?course_id=33624";
var coursePMgu = "https://www.gu.se/studera/hitta-utbildning/bayesiansk-dataanalys-och-maskininlarning-fym285";
var courseScheduleTimeEdit = "https://cloud.timeedit.net/chalmers/web/public/riq80QggY05Zx6Q5Q37y6Y7665Z456X663Z70Z6Q65o60Q6ZY0u0gQnwZq6Qo.html";
