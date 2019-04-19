-- phpMyAdmin SQL Dump
-- version 4.5.4.1deb2ubuntu2.1
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Apr 20, 2019 at 02:07 AM
-- Server version: 5.7.25-0ubuntu0.16.04.2
-- PHP Version: 7.0.33-0ubuntu0.16.04.3

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `shopdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `liveusers`
--

CREATE TABLE `liveusers` (
  `name` varchar(20) NOT NULL,
  `balance` int(20) NOT NULL,
  `items` varchar(50) NOT NULL,
  `email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `liveusers`
--

INSERT INTO `liveusers` (`name`, `balance`, `items`, `email`) VALUES
('John', 22, 'apple', 'john@example.com');

-- --------------------------------------------------------

--
-- Table structure for table `temp_qr`
--

CREATE TABLE `temp_qr` (
  `slno` int(20) NOT NULL,
  `usr` varchar(20) NOT NULL,
  `qrvalue` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `temp_qr`
--

INSERT INTO `temp_qr` (`slno`, `usr`, `qrvalue`) VALUES
(2, 'asd', '2342424'),
(3, 'testUSER', '123');

-- --------------------------------------------------------

--
-- Table structure for table `users1`
--

CREATE TABLE `users1` (
  `id` int(10) NOT NULL,
  `name` varchar(10) NOT NULL,
  `pswd` varchar(20) NOT NULL,
  `addr` varchar(10) NOT NULL,
  `balance` int(10) NOT NULL,
  `email` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `users1`
--

INSERT INTO `users1` (`id`, `name`, `pswd`, `addr`, `balance`, `email`) VALUES
(101, 'abhi', '11', 'atl', 200, ''),
(102, 'patti', '12', 'tvm', 500, ''),
(332, 'asd', 'sad', 'rf', 43, 'sadd@sd.com'),
(332, 'asd', 'sad', 'rf', 43, 'sadd@sd.com'),
(332, 'c', 'sdf', 'df', 3455, 'vf@d.v'),
(0, 'dsf', 'w', 'we', 23, 'w@dxm.c'),
(0, 'dsf', 'w', 'we', 23, 'w@dxm.c'),
(332, 'dsf', 'w', 'we', 23, 'w@dxm.c'),
(332, 'dsf', 'w', 'we', 23, 'w@dxm.c'),
(332, 'dsf', 'w', 'we', 23, 'w@dxm.c'),
(332, 'as', 'ade', 'slkdj', 9, 'a2@a.com'),
(332, 'sd', 'as', 'as', 12, 'sd@s.x'),
(332, 'sd111', 'as', 'xf', 321, 'bafasu@digital-work.net'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'sd', 'sd', 'dc', 23, 'saad@md.dsd'),
(332, 'zero', 'dff', 'a', 0, 'a2@a.com');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `temp_qr`
--
ALTER TABLE `temp_qr`
  ADD UNIQUE KEY `slno` (`slno`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `temp_qr`
--
ALTER TABLE `temp_qr`
  MODIFY `slno` int(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=8;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
